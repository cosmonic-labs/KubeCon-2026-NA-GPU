#!/usr/bin/env node
// Re-times a raw asciinema v2 recording of scripts/walkthrough.sh for playback.
//
//   node scripts/retime.mjs <raw.cast> <out.cast> [--type-ms 20] [--max-gap 3]
//
// The walkthrough runs at full speed, so the raw recording takes only as long as the real work.
// This adds the presentation timing afterwards:
//
// - A command between \e]9999;type\a and \e]9999;/type\a is typed out, one character every
//   --type-ms milliseconds.
// - \e]9999;hold=N\a pauses N seconds, for reading the comment before it.
// - A gap of more than --max-gap seconds between two events, such as a download or a build,
//   is cut to --max-gap.
//
// A command or a marker can be split across two events, as the terminal reads output in chunks.
// It fails if a marker is left over at the end, or if a timestamp would go backwards.

import { readFileSync, writeFileSync } from "node:fs";

const usage = "usage: retime.mjs <raw.cast> <out.cast> [--type-ms 20] [--max-gap 3]";
const args = process.argv.slice(2);
const options = { "--type-ms": 20, "--max-gap": 3 };
const files = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] in options) {
    options[args[i]] = Number(args[++i]);
    if (!Number.isFinite(options[args[i - 1]])) throw new Error(usage);
  } else {
    files.push(args[i]);
  }
}
if (files.length !== 2) throw new Error(usage);
const [input, output] = files;
const typeStep = options["--type-ms"] / 1000;
const maxGap = options["--max-gap"];

const MARKER = /\x1b\]9999;(type|\/type|hold=([0-9.]+))\x07/g;

const lines = readFileSync(input, "utf8").split("\n").filter((line) => line.length > 0);
const header = JSON.parse(lines[0]);
delete header.idle_time_limit;

const out = [];
let rawPrev = 0;
let now = 0;
let typing = false;
// The start of a marker at the end of an event, held for the next one.
let carry = "";
const MARKER_START = "\x1b]9999;";

const emit = (text) => {
  if (text.length === 0) return;
  out.push([Number(now.toFixed(6)), "o", text]);
};

for (const [n, line] of lines.slice(1).entries()) {
  const [time, kind, data] = JSON.parse(line);
  if (time < rawPrev) throw new Error(`event ${n + 1}: time ${time} is before ${rawPrev}`);
  now += Math.min(time - rawPrev, maxGap);
  rawPrev = time;
  if (kind !== "o") {
    out.push([Number(now.toFixed(6)), kind, data]);
    continue;
  }

  let rest = carry + data;
  carry = "";
  const open = rest.lastIndexOf("\x1b");
  if (open !== -1) {
    const tail = rest.slice(open);
    const partial = tail.length < MARKER_START.length
      ? MARKER_START.startsWith(tail)
      : tail.startsWith(MARKER_START) && !tail.includes("\x07");
    if (partial) {
      carry = tail;
      rest = rest.slice(0, open);
    }
  }
  let match;
  MARKER.lastIndex = 0;
  let last = 0;
  while ((match = MARKER.exec(rest)) !== null) {
    const before = rest.slice(last, match.index);
    if (typing) {
      for (const char of before) {
        now += typeStep;
        emit(char);
      }
    } else {
      emit(before);
    }
    last = match.index + match[0].length;
    if (match[1] === "type") {
      if (typing) throw new Error(`event ${n + 1}: typing marker inside a typed command`);
      typing = true;
    } else if (match[1] === "/type") {
      if (!typing) throw new Error(`event ${n + 1}: end of typing without a start`);
      typing = false;
    } else {
      if (typing) throw new Error(`event ${n + 1}: hold marker inside a typed command`);
      now += Number(match[2]);
    }
  }
  const after = rest.slice(last);
  if (typing) {
    for (const char of after) {
      now += typeStep;
      emit(char);
    }
  } else {
    emit(after);
  }
}
if (carry) throw new Error("the recording ends inside a marker");
if (typing) throw new Error("the recording ends inside a typed command");

let prev = 0;
for (const [time, , data] of out) {
  if (time < prev) throw new Error(`output time ${time} is before ${prev}`);
  if (data.includes("\x1b]9999;")) throw new Error(`a marker is left over at ${time}`);
  prev = time;
}

writeFileSync(output, [JSON.stringify(header), ...out.map((event) => JSON.stringify(event))].join("\n") + "\n");
console.log(`${input}: ${lines.length - 1} events, ${rawPrev.toFixed(1)} s recorded`);
console.log(`${output}: ${out.length} events, ${prev.toFixed(1)} s to play`);
