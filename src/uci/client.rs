use shakmaty::{
    Position,
    Chess, 
    Move, 
    Color,
    CastlingMode,
    fen::{Fen},
    uci::{UciMove}
};
use vampirc_uci::parse_with_unknown;
use vampirc_uci::{
    UciMessage, 
    MessageList, 
    UciTimeControl, 
    UciOptionConfig
};
use::vampirc_uci;
use crate::engine::{
    ChessEngine, 
    SearchHandle,
    SearchProgress, 
    SearchResult,
    SearchEngine,
    Limits,
    Score,
    TimeMode,
    Clock,
};
use chrono;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::ops::ControlFlow;

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
    Clear,
}


struct Message {
    text: String,
    msg: UciMessage
}


struct MessageHandler {
    position: Chess,
    limits: Option<Limits>
}


pub fn run() {
    if let Err(e) = run_with(BufReader::new(io::stdin()), io::stdout().lock()) {
        eprintln!("uci: {e}");
    }
}

pub fn run_with<R: BufRead + Send + 'static, W: Write>(input: R, mut output: W) -> io::Result<()>{
    let (tx, rx) = mpsc::channel::<Event>();
    let (e_tx, e_rx) = mpsc::channel::<EMessage>();
    let line_tx: Sender<Event> = tx.clone();
    let en_tx: Sender<Event> = tx.clone();
    thread::spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else {continue;};
            // A closed channel means the main loop is shutting down, not an error.
            if line_tx.send(Event::Line(line)).is_err() {
                break;
            }
        }
    }); 
    let mut engine = SearchEngine::new(16);
    let mut search_handle = engine.search_handle();
    thread::spawn(move || {
        for msg in e_rx {
            match msg {
                EMessage::Search => {
                    let res = engine.search(&mut |p| {
                        let _ = en_tx.send(Event::Info(format_progress(p)));
                    });
                    if en_tx.send(Event::SearchDone(res)).is_err() {
                        break;
                    }
                },
                EMessage::SetHashSize(size_mb) => engine.set_hash_size_mb(size_mb),
                EMessage::SetPosition(pos) => engine.set_position(pos),
                EMessage::PlayMove(mv) => { let _ = engine.play_move(mv); },
                EMessage::Clear => {
                    engine.clear();
                }
            }
        }
    });
    let mut msg_handler = MessageHandler::new();
    'main: for event in rx.iter() {
        match event {
            Event::Line(line) => {
                let messages: MessageList = parse_with_unknown(line.as_str());
                for uci_msg in messages.iter() {
                    let msg = Message {
                        text: line.clone(), 
                        msg: uci_msg.clone()
                    };
                    match msg_handler.handle(&msg, &mut output, &mut search_handle, &e_tx)? {
                        ControlFlow::Continue(_) => {},
                        ControlFlow::Break(_) => break 'main
                    }
                }
            },
            Event::Info(line) => {
                writeln!(output, "{}", line)?;
                output.flush()?;
            },
            Event::SearchDone(res) => { 
                match res.best_move {
                    Some(mv) => {
                        let ponder = if res.pv.len() > 1 {
                            format!(" ponder {}", res.pv[1].to_uci(CastlingMode::Standard))
                        } else {
                            "".to_string()
                        };
                        writeln!(output, "bestmove {}{ponder}", mv.to_uci(CastlingMode::Standard))?;
                    },
                    None => writeln!(output, "bestmove 0000")?,
                }
                output.flush()?;
            }
        }
    }
    Ok(())
}

impl MessageHandler {

    fn new() -> MessageHandler {
        MessageHandler {
            position: Chess::default(),
            limits: Some(Limits::default())
        }
    }

    fn handle<W: Write>(
        &mut self,
        message: &Message, 
        output: &mut W,
        handle: &mut SearchHandle,
        e_tx: &Sender<EMessage>
    ) -> io::Result<ControlFlow<()>>{
        match &message.msg {
            UciMessage::Unknown(_, _) => {},
            UciMessage::Uci => {
                writeln!(output, "{}", UciMessage::id_name("Chess-engine"))?;
                writeln!(output, "{}", UciMessage::id_author("Jesper"))?;
                writeln!(output, "{}", UciMessage::Option(UciOptionConfig::Spin { name: "Hash".to_string(), default: Some(16), min: Some(1), max: Some(4096) }))?;
                writeln!(output, "{}", UciMessage::Option(UciOptionConfig::Check { name: "Ponder".to_string(), default: Some(true) }))?;
                writeln!(output, "{}", UciMessage::UciOk)?;
                output.flush()?;
            },
            UciMessage::SetOption { name, value } => {
                match name.as_str() {
                    "Hash" => {
                        if let Some(value_str) = value {
                            if let Ok(size) = value_str.trim().parse::<usize>() {
                                send_to_engine(e_tx, EMessage::SetHashSize(size.clamp(1, 4096)))?;
                            };
                        } 
                    },
                    _ => {}
                }
            }
            UciMessage::IsReady => {
                writeln!(output, "{}", UciMessage::ReadyOk)?;
                output.flush()?;
            },
            UciMessage::Quit => {
                handle.stop();
                return Ok(ControlFlow::Break(())) 
            }
            UciMessage::Position { startpos, fen, moves } => {
                let mut pos: Chess = if *startpos {
                    Chess::default()
                } else {
                    let Some(f) = fen else { return Ok(ControlFlow::Continue(())) };
                    let Ok(parsed) = f.as_str().parse::<Fen>() else { return Ok(ControlFlow::Continue(())) };
                    let Ok(p) = parsed.into_position(CastlingMode::Standard) else { return Ok(ControlFlow::Continue(())) };
                    p
                };
                for um in moves {
                    let Some(mv) = convert_move(um, &pos) else { break };
                    pos.play_unchecked(mv);
                }
                send_to_engine(e_tx, EMessage::SetPosition(pos.clone()))?;
                self.position = pos;
            },
            UciMessage::Stop => {
                handle.stop();
            },
            UciMessage::UciNewGame => {
                send_to_engine(e_tx, EMessage::Clear)?;
            },
            UciMessage::PonderHit => {
                if let Some(l) = self.limits.take() {
                    handle.set_limits(l);
                }
            },
            UciMessage::Go { time_control, search_control } => {
                let is_ponder = message.text.split_whitespace().any(|w| w == "ponder");
                let mut limits = Limits::default();
                if let Some(t_ctl) = time_control {
                    match t_ctl {
                        UciTimeControl::Infinite => { limits.time_mode = TimeMode::Unbound },
                        UciTimeControl::TimeLeft { 
                            white_time, 
                            black_time, 
                            white_increment, 
                            black_increment, 
                            moves_to_go 
                        } => {
                            let black_to_move = self.position.turn() == Color::Black;
                            let (mine, theirs, increment) = if black_to_move {
                                (black_time, white_time, black_increment)
                            } else {
                                (white_time, black_time, white_increment)
                            };
                            limits.time_mode = match mine {
                                Some(remaining) => TimeMode::Clock(Clock {
                                    remaining: td_to_std(*remaining),
                                    opp_remaining: theirs.map(td_to_std).unwrap_or_default(),
                                    increment: increment.map(td_to_std).unwrap_or_default(),
                                    moves_to_go: moves_to_go.map(u32::from),
                                }),
                                None => TimeMode::Unbound,
                            };
                        }
                        UciTimeControl::Ponder => { limits.time_mode = TimeMode::Unbound },
                        UciTimeControl::MoveTime(dur) => { limits.time_mode = TimeMode::Fixed(td_to_std(*dur))}
                    }
                }

                if let Some(s_ctl) = search_control {
                    limits.max_depth = s_ctl.depth;
                    limits.max_nodes = s_ctl.nodes;
                    let moves: Vec<Move> = s_ctl.search_moves
                        .iter()
                        .filter_map( |mv| convert_move(mv, &self.position))
                        .collect();
                    limits.restrict_to = (!moves.is_empty()).then_some(moves);

                }
                handle.reset();
                self.limits = Some(limits.clone()); 
                if is_ponder {
                    handle.set_limits(limits.with_mode(TimeMode::Unbound));
                } else {
                    handle.set_limits(limits);
                }
                send_to_engine(e_tx, EMessage::Search)?;
            },
            _ => {},
        }
        Ok(ControlFlow::Continue(()))
    }

}

fn send_to_engine(e_tx: &Sender<EMessage>, msg: EMessage) -> io::Result<()> {
    e_tx.send(msg)
        .map_err(|_| io::Error::other("engine thread stopped"))
}

// Converts UciMove to shakmaty Move
fn convert_move(uci_move: &vampirc_uci::UciMove, pos: &Chess) -> Option<Move> {
    let Ok(uci) = uci_move.to_string().parse::<UciMove>() else { return None };
    let Ok(mv) = uci.to_move(pos) else { return None };
    Some(mv)
}


fn td_to_std(td: chrono::Duration) -> std::time::Duration {
    std::time::Duration::from_millis(td.num_milliseconds().max(0) as u64)
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



