//! Converts datagen text into the binary records `nnue/train.py` reads.
//!
//!   cargo run --release --bin pack -- data.txt data.bin
//!
//! Python is slow at parsing FENs, so this is done once, here. Each position
//! becomes one fixed-size 68-byte record (all little-endian):
//!
//!   features  [u16; 32]  board features from White's point of view, see below;
//!                        unused slots hold 768
//!   score     i16        search score in centipawns, White's point of view
//!   result    u8         0 Black won, 1 draw, 2 White won
//!   stm       u8         side to move: 0 White, 1 Black
//!
//! A feature is one piece on one square: `color * 384 + role * 64 + square`,
//! with color 0 = White, role 0..6 = pawn..king and square 0..64 = a1..h8.
//! Black's point of view and the flip to the side to move are left to the
//! trainer, so this file only ever describes the board as it stands.

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, Position};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::exit;

/// Fills the unused feature slots; one past the last real feature.
const NO_FEATURE: u16 = 768;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        eprintln!("usage: pack <data.txt> <data.bin>");
        exit(2)
    };
    let reader = BufReader::new(File::open(input).unwrap_or_else(|e| fail(&format!("cannot open {input}: {e}"))));
    let mut out = BufWriter::new(File::create(output).unwrap_or_else(|e| fail(&format!("cannot create {output}: {e}"))));

    let (mut written, mut skipped) = (0u64, 0u64);
    for (n, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| fail(&format!("read error: {e}")));
        if line.trim().is_empty() {
            continue;
        }
        match record(&line) {
            Some(bytes) => {
                out.write_all(&bytes).unwrap_or_else(|e| fail(&format!("write error: {e}")));
                written += 1;
            }
            None => {
                if skipped < 10 {
                    eprintln!("line {}: cannot parse, skipped: {line}", n + 1);
                }
                skipped += 1;
            }
        }
    }
    out.flush().unwrap_or_else(|e| fail(&format!("write error: {e}")));
    println!("{written} positions written to {output}, {skipped} lines skipped");
}

/// One `fen | score | result` line as a 68-byte record.
fn record(line: &str) -> Option<[u8; 68]> {
    let mut parts = line.split('|').map(str::trim);
    let (fen, score, result) = (parts.next()?, parts.next()?, parts.next()?);
    let pos: Chess = fen.parse::<Fen>().ok()?.into_position(CastlingMode::Standard).ok()?;
    let score: i16 = score.parse::<i32>().ok()?.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let result: u8 = match result {
        "0.0" | "0" => 0,
        "0.5" => 1,
        "1.0" | "1" => 2,
        _ => return None,
    };

    let mut bytes = [0u8; 68];
    let mut features = [NO_FEATURE; 32];
    for (slot, (sq, piece)) in features.iter_mut().zip(pos.board().clone()) {
        let color = if piece.color == Color::White { 0 } else { 1 };
        *slot = color * 384 + (piece.role as u16 - 1) * 64 + sq as u16;
    }
    for (i, f) in features.iter().enumerate() {
        bytes[2 * i..2 * i + 2].copy_from_slice(&f.to_le_bytes());
    }
    bytes[64..66].copy_from_slice(&score.to_le_bytes());
    bytes[66] = result;
    bytes[67] = if pos.turn() == Color::White { 0 } else { 1 };
    Some(bytes)
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    exit(1)
}
