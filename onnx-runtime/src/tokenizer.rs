//! The CLIP byte-level BPE tokenizer, enough for SD-Turbo's text encoder.

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;

const START_OF_TEXT: i32 = 49406;
const END_OF_TEXT: i32 = 49407;
// Padding id the SD-Turbo web example uses.
const PAD: i32 = 0;

pub struct Tokenizer {
    vocab: HashMap<String, i32>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    pattern: Regex,
}

impl Tokenizer {
    /// Loads `vocab.json` and `merges.txt` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name)).map_err(|e| format!("reading {}: {e}", dir.join(name).display()))
        };

        let vocab: HashMap<String, i32> =
            serde_json::from_str(&read("vocab.json")?).map_err(|e| format!("parsing vocab.json: {e}"))?;
        let merge_ranks = read("merges.txt")?
            .lines()
            .filter(|line| !line.starts_with("#version") && !line.is_empty())
            .enumerate()
            .filter_map(|(rank, line)| {
                let (a, b) = line.split_once(' ')?;
                Some(((a.to_string(), b.to_string()), rank))
            })
            .collect();

        Ok(Self {
            vocab,
            merge_ranks,
            byte_to_char: byte_to_char(),
            pattern: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+").unwrap(),
        })
    }

    /// Returns exactly `length` token ids: start, the text (truncated to fit), end, then padding.
    pub fn encode(&self, text: &str, length: usize) -> Vec<i32> {
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();

        let mut ids = vec![START_OF_TEXT];
        for word in self.pattern.find_iter(&text) {
            let word: String = word.as_str().bytes().map(|b| self.byte_to_char[b as usize]).collect();
            ids.extend(self.bpe(&word).iter().filter_map(|token| self.vocab.get(token)));
        }
        ids.truncate(length - 1);
        ids.push(END_OF_TEXT);
        ids.resize(length, PAD);
        ids
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(String::from).collect();
        if let Some(last) = parts.last_mut() {
            last.push_str("</w>");
        }

        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| Some((self.merge_ranks.get(&(pair[0].clone(), pair[1].clone()))?, i)))
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
        parts
    }
}

/// GPT-2's reversible mapping from bytes to printable characters.
fn byte_to_char() -> [char; 256] {
    let printable = |b: u32| matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
    let mut table = ['\0'; 256];
    let mut next = 256;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    table
}
