//! The SD-Turbo pipeline: text encoder, a single UNet step and the VAE decoder.
//!
//! The models are https://huggingface.co/schmuell/sd-turbo-ort-web, the pipeline follows
//! https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/sd-turbo.

use std::path::Path;
use std::time::Instant;

use half::f16;

use crate::onnxruntime::onnxruntime::inference_session::{ExecutionProvider, Session, SessionOptions};
use crate::onnxruntime::onnxruntime::types::{Error, Tensor, TensorElementType};
use crate::tokenizer::Tokenizer;

const SEQUENCE_LENGTH: usize = 77;
const LATENT_SHAPE: [i64; 4] = [1, 4, 64, 64];
// The first sigma of the Euler ancestral scheduler, for SD-Turbo's single step at timestep 999.
const SIGMA: f32 = 14.6146;
const VAE_SCALING_FACTOR: f32 = 0.18215;

pub struct Pipeline {
    tokenizer: Tokenizer,
    text_encoder: Session,
    unet: Session,
    vae_decoder: Session,
}

impl Pipeline {
    pub async fn load(models: &Path) -> Result<Self, String> {
        Ok(Self {
            tokenizer: Tokenizer::load(&models.join("tokenizer"))?,
            text_encoder: load(models, "text_encoder", &[("batch_size", 1)]).await?,
            unet: load(
                models,
                "unet",
                &[("batch_size", 1), ("num_channels", 4), ("height", 64), ("width", 64), ("sequence_length", 77)],
            )
            .await?,
            vae_decoder: load(
                models,
                "vae_decoder",
                &[("batch_size", 1), ("num_channels_latent", 4), ("height_latent", 64), ("width_latent", 64)],
            )
            .await?,
        })
    }

    /// Generates a 512x512 image for `prompt` and returns it as PNG.
    pub async fn generate(&self, prompt: &str, seed: u64) -> Result<Vec<u8>, String> {
        let start = Instant::now();
        let input_ids = self.tokenizer.encode(prompt, SEQUENCE_LENGTH);
        let input_ids = Tensor::from_data(
            TensorElementType::Int32,
            vec![1, SEQUENCE_LENGTH as i64],
            input_ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
        )
        .await
        .map_err(fmt)?;
        let mut outputs =
            self.text_encoder.run(vec![("input_ids".into(), &input_ids)], None, None).await.map_err(fmt)?;
        let (_, hidden_states) = outputs.remove(0);
        let text_encoder_ms = elapsed_ms(start);

        let start = Instant::now();
        let latents = random_normal(LATENT_SHAPE.iter().product::<i64>() as usize, seed, SIGMA);
        let divisor = (SIGMA * SIGMA + 1.0).sqrt();
        let scaled: Vec<f32> = latents.iter().map(|x| x / divisor).collect();
        let sample = float_tensor(input_type(&self.unet, "sample").await, &LATENT_SHAPE, &scaled).await?;
        let timestep =
            Tensor::from_data(TensorElementType::Int64, vec![1], 999i64.to_le_bytes().to_vec()).await.map_err(fmt)?;
        let mut outputs = self
            .unet
            .run(
                vec![
                    ("sample".into(), &sample),
                    ("timestep".into(), &timestep),
                    ("encoder_hidden_states".into(), &hidden_states),
                ],
                None,
                None,
            )
            .await
            .map_err(fmt)?;
        let noise = read_floats(&outputs.remove(0).1).await?;
        let unet_ms = elapsed_ms(start);

        // One Euler step from SIGMA to 0 leaves the predicted clean latents, which the VAE expects unscaled.
        let denoised: Vec<f32> =
            latents.iter().zip(&noise).map(|(x, n)| (x - SIGMA * n) / VAE_SCALING_FACTOR).collect();

        let start = Instant::now();
        let latent_sample =
            float_tensor(input_type(&self.vae_decoder, "latent_sample").await, &LATENT_SHAPE, &denoised).await?;
        let mut outputs =
            self.vae_decoder.run(vec![("latent_sample".into(), &latent_sample)], None, None).await.map_err(fmt)?;
        let (_, image) = outputs.remove(0);
        let dims = image.get_dims().await;
        let pixels = read_floats(&image).await?;
        let vae_decoder_ms = elapsed_ms(start);

        eprintln!(
            "generated {prompt:?} (seed {seed}): text encoder {text_encoder_ms:.0} ms, unet {unet_ms:.0} ms, \
             vae decoder {vae_decoder_ms:.0} ms"
        );
        encode_png(&pixels, dims[3] as u32, dims[2] as u32)
    }
}

async fn load(models: &Path, name: &str, dimensions: &[(&str, i64)]) -> Result<Session, String> {
    let start = Instant::now();
    let options = SessionOptions {
        execution_providers: vec![ExecutionProvider { name: "webgpu".into(), options: vec![] }],
        graph_optimization_level: None,
        execution_mode: None,
        intra_op_num_threads: None,
        free_dimension_overrides: dimensions.iter().map(|(name, size)| (name.to_string(), *size)).collect(),
        log_id: None,
        log_severity_level: None,
        // Same as the web example: keep the weights on the GPU only, instead of also in wasm memory.
        config_entries: vec![
            ("session.disable_prepacking".into(), "1".into()),
            ("session.use_device_allocator_for_initializers".into(), "1".into()),
        ],
    };
    let path = models.join(name).join("model.onnx");
    let session = Session::create_from_path(path.to_string_lossy().into_owned(), Some(options)).await.map_err(fmt)?;
    eprintln!("loaded {name} in {:.1} s", start.elapsed().as_secs_f64());
    Ok(session)
}

async fn input_type(session: &Session, name: &str) -> TensorElementType {
    session
        .get_input_metadata()
        .await
        .into_iter()
        .find(|input| input.name == name)
        .and_then(|input| input.element_type)
        .unwrap_or(TensorElementType::Float32)
}

async fn float_tensor(element_type: TensorElementType, dims: &[i64], values: &[f32]) -> Result<Tensor, String> {
    let bytes: Vec<u8> = match element_type {
        TensorElementType::Float16 => values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect(),
        _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    Tensor::from_data(element_type, dims.to_vec(), bytes).await.map_err(fmt)
}

async fn read_floats(tensor: &Tensor) -> Result<Vec<f32>, String> {
    let bytes = tensor.get_data().await.map_err(fmt)?;
    Ok(match tensor.get_element_type().await {
        TensorElementType::Float16 => {
            bytes.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()
        }
        _ => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
    })
}

/// Encodes a CHW image with values in [-1, 1] as an RGB PNG.
fn encode_png(chw: &[f32], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let plane = (width * height) as usize;
    let rgb: Vec<u8> = (0..plane)
        .flat_map(|i| (0..3).map(move |c| ((chw[c * plane + i] / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect();

    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(&rgb).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(png)
}

/// Normally distributed values with standard deviation `sigma`, from xorshift64* and the Box-Muller transform.
fn random_normal(count: usize, seed: u64, sigma: f32) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut uniform = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        // 53 random bits in (0, 1]
        ((state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) + 1) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32 * sigma
        })
        .collect()
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn fmt(e: Error) -> String {
    format!("{:?}: {}", e.code, e.message)
}
