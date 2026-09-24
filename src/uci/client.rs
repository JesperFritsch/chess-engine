use shakmaty::{
    Position,
    Chess, 
    Move, 
    CastlingMode,
    fen::{Fen},
    uci::{UciMove}
};
use vampirc_uci::parse_with_unknown;
use vampirc_uci::{
    UciMessage, 
    MessageList, 
    UciTimeControl, 
    Serializable,
};
use::vampirc_uci;
use crate::engine::{
    ChessEngine, 
    SearchHandle,
    SearchProgress, 
    SearchResult,
    SearchEngine,
    Limits,
    IllegalMove,
    Score
};
use std::io::{self, BufRead, BufReader, Write};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::str::FromStr;

pub enum Event {
    Line(String),
    Info(String),
    SearchDone(SearchResult),
}

pub enum EMessage {
    Search,
    SetHashSize(usize),
    SetPosition(Chess),
    PlayMove(Move),
}


struct Message {
    text: String,
    msg: UciMessage
}


struct MessageHandler {
    position: Option<Chess>
}


pub fn run() {
    run_with(BufReader::new(io::stdin()), io::stdout().lock());
}

pub fn run_with<R: BufRead + Send + 'static, W: Write>(input: R, mut output: W) -> io::Result<()>{
    let (tx, rx) = mpsc::channel::<Event>();
    let (e_tx, e_rx) = mpsc::channel::<EMessage>();
    let line_tx: Sender<Event> = tx.clone();
    let en_tx: Sender<Event> = tx.clone();
    thread::spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else {continue;};
            line_tx.send(Event::Line(line)).unwrap();
        }
    }); 
    let mut engine = SearchEngine::new(500);
    let mut search_handle = engine.search_handle();
    thread::spawn(move || {
        for msg in e_rx {
            match msg {
                EMessage::Search => {
                    let res = engine.search(&mut |p| {
                        en_tx.send(Event::Info(format_progress(p)));
                    });
                    en_tx.send(Event::SearchDone(res));      
                },
                EMessage::SetHashSize(size_mb) => engine.set_hash_size_mb(size_mb),
                EMessage::SetPosition(pos) => engine.set_position(pos),
                EMessage::PlayMove(mv) => { let _ = engine.play_move(mv); }
            }
        }
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
            },
            Event::Info(line) => {
                writeln!(output, "{}", line);
                output.flush()?;
            },
            Event::SearchDone(res) => { 
                match res.best_move {
                    Some(mv) => writeln!(output, "bestmove {}", mv.to_uci(CastlingMode::Standard))?,
                    None => writeln!(output, "bestmove 0000")?,
                }
                output.flush()?;
            }
        }
    }
    Ok(())
}

impl MessageHandler {

    fn handle<W: Write>(
        &mut self,
        message: &Message, 
        output: &mut W,
        handle: &mut SearchHandle,
        e_tx: &mut Sender<EMessage>
    ) -> io::Result<()>{
        match &message.msg {
            UciMessage::Unknown(_, _) => {},
            UciMessage::IsReady => {
                writeln!(output, "{}", UciMessage::ReadyOk)?;
                output.flush()?;
            },
            UciMessage::Position { startpos, fen, moves } => {
                let mut pos: Chess = if *startpos {
                    Chess::default()
                } else {
                    let Some(f) = fen else { return Ok(()) };
                    let Ok(parsed) = f.as_str().parse::<Fen>() else { return Ok(()) };
                    let Ok(p) = parsed.into_position(CastlingMode::Standard) else { return Ok(()) };
                    p
                };
                for um in moves {
                    let Some(mv) = convert_move(um, &pos) else { break };
                    pos.play_unchecked(mv);
                }
                e_tx.send(EMessage::SetPosition(pos.clone()));
                self.position = Some(pos);
            },
            UciMessage::Stop => {},
            UciMessage::UciNewGame => {},
            UciMessage::PonderHit => {},
            UciMessage::Go { time_control, search_control } => {
                let is_ponder = message.text.split_whitespace().any(|w| w == "ponder");
                let limits = Limits::default();
            },
            _ => {},
        }
        Ok(())
    }

}

// Converts UciMove to shakmaty Move
fn convert_move(uci_move: &vampirc_uci::UciMove, pos: &Chess) -> Option<Move> {
    let Ok(uci) = uci_move.to_string().parse::<UciMove>() else { return None };
    let Ok(mv) = uci.to_move(pos) else { return None };
    Some(mv)
}


fn format_progress(p: &SearchProgress) -> String {
    let mut s = format!("info depth {} seldepth {}", p.depth, p.seldepth);
    match p.score {
        Score::Cp(cp) => s.push_str(&format!(" score cp {cp}")),
        Score::Mate(n) => s.push_str(&format!(" score mate {n}")),
    }
    s.push_str(&format!(
        " nodes {} nps {} hashfull {} time {}",
        p.nodes, p.nodes_per_s, p.hashfull, p.elapsed.as_millis()
    ));
    if !p.pv.is_empty() {
        s.push_str(" pv");
        for m in p.pv {
            s.push_str(&format!(" {}", m.to_uci(CastlingMode::Standard)));
        }
    }
    s
}

#[test]
fn parse_unknown() {
    let input = "unknown\nanother unknown\n";
    let mut output = Vec::new();
    run_with(input.as_bytes(), &mut output);

    let out = String::from_utf8(output).unwrap();
    assert!(out.len() == 0);
}

