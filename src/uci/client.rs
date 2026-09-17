use vampirc_uci::parse_with_unknown;
use vampirc_uci::{UciMessage, MessageList, UciTimeControl, Serializable};

use std::io::{self, BufRead, Write};

pub fn run() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_with(stdin.lock(), stdout.lock());
}

pub fn run_with<R: BufRead, W: Write>(input: R, mut output: W) {
    for line in input.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let m_list = parse_with_unknown(line.as_str());
        for message in m_list {
            match message {
                UciMessage::Unknown(_, _) => {}
                _ => {}
            }
        }
    }
}



#[test]
fn parse_unknown() {
    let input = "unknown\nanother unknown\n";
    let mut output = Vec::new();
    run_with(input.as_bytes(), &mut output);

    let out = String::from_utf8(output).unwrap();
    assert!(out.len() == 0);
}
