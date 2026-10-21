//! Converts a UCI `position` command to a FEN.
//!
//!   echo "position startpos moves e2e4 e7e5" | uci2fen
//!   uci2fen position fen <fen> moves g1f3
//!
//! Takes the command as arguments, or one command per line on stdin, and
//! prints the FEN of the resulting position, one per line. The leading
//! `position` is optional.

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};
use std::io::{BufRead, Write};
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if !args.is_empty() {
        writeln!(out, "{}", convert_or_exit(&args.join(" "))).unwrap();
        return;
    }
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("read error: {e}");
            exit(1)
        });
        if line.trim().is_empty() {
            continue;
        }
        writeln!(out, "{}", convert_or_exit(&line)).unwrap();
    }
}

fn convert_or_exit(command: &str) -> String {
    convert(command).unwrap_or_else(|e| {
        eprintln!("{e}");
        exit(1)
    })
}

fn convert(command: &str) -> Result<String, String> {
    let mut tokens = command.split_whitespace().peekable();
    if tokens.peek() == Some(&"position") {
        tokens.next();
    }
    let mut pos: Chess = match tokens.next() {
        Some("startpos") => Chess::default(),
        Some("fen") => {
            let fen: Vec<&str> = tokens.by_ref().take_while(|&t| t != "moves").collect();
            let fen = fen.join(" ");
            fen.parse::<Fen>()
                .ok()
                .and_then(|f| f.into_position(CastlingMode::Standard).ok())
                .ok_or_else(|| format!("cannot parse FEN: {fen}"))?
        }
        _ => return Err(format!("expected `startpos` or `fen`: {command}")),
    };
    // After `startpos` the next token is `moves`; after a FEN, take_while
    // already consumed it.
    if tokens.peek() == Some(&"moves") {
        tokens.next();
    }
    for token in tokens {
        let mv = token
            .parse::<UciMove>()
            .ok()
            .and_then(|uci| uci.to_move(&pos).ok())
            .ok_or_else(|| format!("illegal move {token} in position {}", Fen::from_position(&pos, EnPassantMode::Legal)))?;
        pos.play_unchecked(mv);
    }
    Ok(Fen::from_position(&pos, EnPassantMode::Legal).to_string())
}
