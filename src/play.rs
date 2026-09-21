//! Renderer-agnostic game orchestration.
//!
//! The [`play_game`] loop knows nothing about *how* the game is displayed or
//! how a human's move is collected — it talks only to the [`Frontend`] trait.
//! Today the only implementation is the terminal UI in [`crate::tui`], but a
//! different front end (a Lichess bridge, a GUI, a network peer, …) can be
//! dropped in without touching the loop.
//!
//! Likewise the engine is hidden behind [`ChessEngine`], so the loop is
//! agnostic to *who* is thinking on the other side of the board.

use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use shakmaty::{Chess, Color, Move, Position};
use shakmaty::san::SanPlus;
use crate::engine::ChessEngine;

use crate::engine::{time_bound_search_with_progress, SearchContext, SearchInfo};

/// Everything a front end needs to render a single game position.
///
/// It is a plain data snapshot so a front end never has to reach back into the
/// game loop or the engine.
pub struct BoardView {
    /// The position to display.
    pub pos: Chess,
    /// The side the human is playing; used to orient the board.
    pub perspective: Color,
    /// The move that produced `pos`, if any (for highlighting).
    pub last_move: Option<Move>,
    /// A short human-readable status / prompt line.
    pub status: String,
}

/// The outcome of asking a human for a move.
pub enum PlayerMove {
    /// Play this (guaranteed legal) move.
    Play(Move),
    /// The human wants to abandon the game.
    Quit,
}

/// The interface between the game loop and whatever is driving the game for the
/// human player. A front end is responsible for both *input* (collecting the
/// human's move) and *output* (showing the board).
pub trait Frontend {
    /// Ask the human which color to play. `Ok(None)` means "quit before we
    /// start".
    fn choose_side(&mut self) -> io::Result<Option<Color>>;

    /// Block until the human has chosen a move for `view.pos`, or asks to quit.
    fn request_move(&mut self, view: &BoardView) -> io::Result<PlayerMove>;

    /// Passively display a position (e.g. while the engine is thinking).
    fn show(&mut self, view: &BoardView) -> io::Result<()>;

    /// Update the display with the engine's latest completed search depth, in
    /// real time. Called once per depth while the engine thinks. Progress
    /// updates are best-effort, so this returns `()` rather than a `Result`.
    fn thinking(&mut self, view: &BoardView, info: &SearchInfo);

    /// Show the final position and result, offer to save the game as PGN
    /// (`pgn` is the complete, ready-to-write game record), then wait for
    /// acknowledgement.
    fn game_over(&mut self, view: &BoardView, pgn: &str) -> io::Result<()>;
}

/// Drive a full game between the human (via `front`) and the `engine`.
///
/// The loop is deliberately tiny and free of any rendering concerns: it just
/// alternates turns, forwards legal moves into the position, and reports the
/// result. Swap `front` for another [`Frontend`] to change the entire user
/// experience.
pub fn play_game(front: &mut dyn Frontend, engine: &mut dyn ChessEngine) -> io::Result<()> {
    let human = match front.choose_side()? {
        Some(color) => color,
        None => return Ok(()),
    };

    let mut pos = Chess::default();
    let mut last_move: Option<Move> = None;
    // A short note about the previous move, shown above the board.
    let mut last_note: Option<String> = None;
    // The moves played so far, in SAN, for building the PGN at the end.
    let mut history: Vec<String> = Vec::new();

    while !pos.is_game_over() {
        if pos.turn() == human {
            let view = BoardView {
                pos: pos.clone(),
                perspective: human,
                last_move,
                status: prompt_line(&pos, last_note.as_deref()),
            };
            match front.request_move(&view)? {
                PlayerMove::Play(mv) => {
                    let san = SanPlus::from_move(pos.clone(), mv).to_string();
                    last_note = Some(format!("You played {san}."));
                    history.push(san);
                    last_move = Some(mv);
                    pos = pos.play(mv).expect("front end returned a legal move");
                }
                PlayerMove::Quit => return Ok(()),
            }
        } else {
            let view = BoardView {
                pos: pos.clone(),
                perspective: human,
                last_move,
                status: engine_line("Engine is thinking…", last_note.as_deref()),
            };
            front.show(&view)?;
            // Forward each completed depth to the front end for a live display.
            let chosen = engine.best_move(&pos, &mut |info| front.thinking(&view, info));
            match chosen {
                Some(mv) => {
                    let san = SanPlus::from_move(pos.clone(), mv).to_string();
                    last_note = Some(format!("Engine played {san}."));
                    history.push(san);
                    last_move = Some(mv);
                    pos = pos.play(mv).expect("engine returned a legal move");
                }
                None => break,
            }
        }
    }

    let pgn = to_pgn(&history, &pos, human);
    let view = BoardView {
        pos: pos.clone(),
        perspective: human,
        last_move,
        status: result_text(&pos, human),
    };
    front.game_over(&view, &pgn)
}

fn prompt_line(pos: &Chess, last_note: Option<&str>) -> String {
    let mut line = String::new();
    if let Some(note) = last_note {
        line.push_str(note);
        line.push(' ');
    }
    if pos.is_check() {
        line.push_str("Your move — you are in check!");
    } else {
        line.push_str("Your move.");
    }
    line
}

fn engine_line(base: &str, last_note: Option<&str>) -> String {
    match last_note {
        Some(note) => format!("{note} {base}"),
        None => base.to_string(),
    }
}

fn result_text(pos: &Chess, human: Color) -> String {
    use shakmaty::KnownOutcome;
    match pos.outcome().known() {
        Some(KnownOutcome::Decisive { winner }) => {
            if winner == human {
                "Checkmate — you win! 🎉".to_string()
            } else {
                "Checkmate — the engine wins.".to_string()
            }
        }
        Some(KnownOutcome::Draw) => "Draw.".to_string(),
        None => "Game over.".to_string(),
    }
}

/// Build a complete PGN document for a finished game: the seven-tag roster
/// followed by the numbered move text and the result token.
fn to_pgn(history: &[String], pos: &Chess, human: Color) -> String {
    let result = pos.outcome().as_str(); // "1-0" | "0-1" | "1/2-1/2" | "*"
    let (white, black) = match human {
        Color::White => ("Human", "Rust Chess Engine"),
        Color::Black => ("Rust Chess Engine", "Human"),
    };
    let (y, m, d) = today_utc();

    let mut pgn = String::new();
    pgn.push_str("[Event \"Casual game\"]\n");
    pgn.push_str("[Site \"Rust Chess TUI\"]\n");
    pgn.push_str(&format!("[Date \"{y:04}.{m:02}.{d:02}\"]\n"));
    pgn.push_str("[Round \"-\"]\n");
    pgn.push_str(&format!("[White \"{white}\"]\n"));
    pgn.push_str(&format!("[Black \"{black}\"]\n"));
    pgn.push_str(&format!("[Result \"{result}\"]\n"));
    pgn.push('\n');

    let body = movetext(history);
    if body.is_empty() {
        pgn.push_str(result);
    } else {
        pgn.push_str(&body);
        pgn.push(' ');
        pgn.push_str(result);
    }
    pgn.push('\n');
    pgn
}

/// Turn a SAN move list into wrapped PGN move text: `1. e4 e5 2. Nf3 …`.
fn movetext(history: &[String]) -> String {
    let mut tokens: Vec<String> = Vec::new();
    for (i, san) in history.iter().enumerate() {
        if i % 2 == 0 {
            tokens.push(format!("{}.", i / 2 + 1));
        }
        tokens.push(san.clone());
    }

    // Wrap at 80 columns, the conventional PGN line width.
    let mut out = String::new();
    let mut line = String::new();
    for tok in tokens {
        if line.is_empty() {
            line = tok;
        } else if line.len() + 1 + tok.len() > 80 {
            out.push_str(&line);
            out.push('\n');
            line = tok;
        } else {
            line.push(' ');
            line.push_str(&tok);
        }
    }
    out.push_str(&line);
    out
}

/// Current UTC date as `(year, month, day)`, using only the standard library.
fn today_utc() -> (i64, u32, u32) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    civil_from_days(secs.div_euclid(86_400))
}

/// Convert days since the Unix epoch to a civil `(year, month, day)`.
///
/// Howard Hinnant's well-known `civil_from_days` algorithm (public domain).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::san::San;

    /// Play a sequence of SAN moves, returning the SAN history (as the game loop
    /// records it) and the final position.
    fn play_sans(sans: &[&str]) -> (Vec<String>, Chess) {
        let mut pos = Chess::default();
        let mut history = Vec::new();
        for s in sans {
            let san: San = s.parse().unwrap();
            let mv = san.to_move(&pos).unwrap();
            history.push(SanPlus::from_move(pos.clone(), mv).to_string());
            pos = pos.play(mv).unwrap();
        }
        (history, pos)
    }

    #[test]
    fn scholars_mate_pgn() {
        let (history, pos) =
            play_sans(&["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]);
        assert!(pos.is_checkmate());

        let pgn = to_pgn(&history, &pos, Color::White);
        assert!(pgn.contains("[Result \"1-0\"]"), "{pgn}");
        assert!(pgn.contains("[White \"Human\"]"), "{pgn}");
        assert!(pgn.contains("[Black \"Rust Chess Engine\"]"), "{pgn}");
        // Numbered move text with the mate suffix and the trailing result token.
        assert!(
            pgn.contains("1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"),
            "{pgn}"
        );
    }

    #[test]
    fn black_perspective_swaps_the_roster() {
        let (history, pos) = play_sans(&["e4", "e5"]);
        let pgn = to_pgn(&history, &pos, Color::Black);
        assert!(pgn.contains("[White \"Rust Chess Engine\"]"), "{pgn}");
        assert!(pgn.contains("[Black \"Human\"]"), "{pgn}");
    }

    #[test]
    fn empty_game_pgn_is_just_the_result() {
        let pgn = to_pgn(&[], &Chess::default(), Color::White);
        // No moves yet, so the game is still in progress: result token is "*".
        assert!(pgn.trim_end().ends_with('*'), "{pgn}");
    }

    #[test]
    fn movetext_numbers_pairs() {
        let history = ["e4", "e5", "Nf3"].map(String::from);
        assert_eq!(movetext(&history), "1. e4 e5 2. Nf3");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(18_628), (2021, 1, 1));
    }
}
