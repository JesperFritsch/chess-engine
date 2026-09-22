use shakmaty::{Chess, Move};
use vampirc_uci::parse_with_unknown;
use vampirc_uci::{UciMessage, MessageList, UciTimeControl, Serializable};
use crate::engine::{
    ChessEngine, 
    SearchHandle,
    SearchProgress, 
    SearchResult,
    SearchEngine
};

use std::io::{self, BufRead, BufReader, Write};
use std::sync::mpsc::{self, Sender};
use std::thread;


pub enum Event {
    Line(String),
}

pub enum EMessage {
    Search,
    SetLimits,
    SetHashSize,
    SetPosition(Chess),
    PlayMove(Move),
}

pub enum EResponse {
    BestMove(Move)
}

struct Message {
    text: String,
    msg: UciMessage
}

pub fn run() {
    run_with(BufReader::new(io::stdin()), io::stdout().lock());
}

pub fn run_with<R: BufRead + Send + 'static, W: Write>(input: R, mut output: W) {
    let (tx, rx) = mpsc::channel::<Event>();
    let (e_tx, e_rx) = mpsc::channel::<EMessage>();
    let (c_tx, c_rx) = mpsc::channel::<EMessage>();
    let line_tx: Sender<Event> = tx.clone();
    thread::spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else {continue;};
            line_tx.send(Event::Line(line)).unwrap();
        }
    }); 
    let engine = SearchEngine::new(500);
    let mut search_handle = engine.search_handle();
    thread::spawn(move || {
        
    });
    for event in rx.iter() {
        match event {
            Event::Line(line) => {
                let messages: MessageList = parse_with_unknown(line.as_str());
                for uci_msg in messages.iter() {
                    let msg = Message {
                        text: line.clone(), 
                        msg: uci_msg.clone()
                    };
                    handle_message(&msg, &mut output).unwrap();
                }
            }
        }
    }
}


fn handle_message<W: Write>(message: &Message, output: &mut W ) -> io::Result<()>{
    match &message.msg {
        UciMessage::Unknown(_, _) => {},
        UciMessage::IsReady => {
            writeln!(output, "{}", UciMessage::ReadyOk)?;
            output.flush()?;
        },
        UciMessage::Position { startpos, fen, moves } => {},
        UciMessage::Stop => {},
        UciMessage::UciNewGame => {},
        UciMessage::PonderHit => {},
        UciMessage::Go { time_control, search_control } => {
            let is_ponder = message.text.split_whitespace().any(|w| w == "ponder");
        },
        _ => {},
    }
    Ok(())
}


#[test]
fn parse_unknown() {
    let input = "unknown\nanother unknown\n";
    let mut output = Vec::new();
    run_with(input.as_bytes(), &mut output);

    let out = String::from_utf8(output).unwrap();
    assert!(out.len() == 0);
}
