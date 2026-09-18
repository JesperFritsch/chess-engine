use vampirc_uci::parse_with_unknown;
use vampirc_uci::{UciMessage, MessageList, UciTimeControl, Serializable};
use crate::engine::{ChessEngine, SearchControl, SearchHandle, SearchProgress, SearchResult};

use std::io::{self, BufRead, BufReader, Write};
use std::sync::mpsc::{self, Sender};
use std::thread;


pub enum Event {
    Line(String),
}

pub fn run() {
    run_with(BufReader::new(io::stdin()), io::stdout().lock());
}

pub fn run_with<R: BufRead + Send + 'static, W: Write>(input: R, mut output: W) {
    let (tx, rx) = mpsc::channel::<Event>();
    let line_tx: Sender<Event> = tx.clone();
    thread::spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else {break};
        }
    })
}



#[test]
fn parse_unknown() {
    let input = "unknown\nanother unknown\n";
    let mut output = Vec::new();
    run_with(input.as_bytes(), &mut output);

    let out = String::from_utf8(output).unwrap();
    assert!(out.len() == 0);
}
