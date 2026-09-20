use shakmaty::{Move, Chess};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};


pub enum Score {
    Cp(i32), // Score in centipawns
    Mate(i32), // mate in x moves, negative if getting mated
}


pub struct Clock {
    pub remaining: Duration,
    pub opp_remaining: Duration,
    pub increment: Duration,
    pub moves_to_go: Option<u32>,
    pub running: bool,
}


pub struct Limits {
    pub deadline: Option<Instant>,
    pub max_depth: Option<u32>,
    pub max_nodes: Option<u64>,
    pub restrict_to: Option<Vec<Move>>,
}


pub struct SearchProgress<'a> {
    pub depth: u32, // depth in plies
    pub seldepth: u32, // absolute max depth, even for extensions and quiescense
    pub elapsed: Duration, // time searched in ms
    pub nodes: u64,
    pub pv: &'a [Move],
    pub score: Score,
    pub hashfull: u32, // hash full in permill (0 - 1000)
    pub nodes_per_s: u32,
}


pub struct SearchResult {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub score: Score,
    pub depth: u32,
    pub nodes: u64,
}


pub struct SearchControl {
    pub clock: Clock,
    pub limits: Limits,
    pub ponder: bool, // if the engine is able to search prospect moves, while opponents turn.
    
}


pub struct Options {
    pub hash_size_mb: u32, // MB size of the hash table
}


#[derive(Clone)]
pub struct SearchHandle{
    pub stop: Arc<AtomicBool>,
    pub control: Arc<Mutex<SearchControl>>,
}

impl SearchHandle {
    pub fn stop(&self) {self.stop.store(true, Ordering::Relaxed)}
    pub fn reset(&self) {self.stop.store(false, Ordering::Relaxed)}
    pub fn is_stopped(&self) -> bool {self.stop.load(Ordering::Relaxed)}
    pub fn set_control(&self, control: SearchControl) {
        *self.control.lock().unwrap() = control;
    }
}

pub trait ChessEngine {
    fn set_position(
        &mut self, 
        pos: Chess
    );

    fn play_move(
        &mut self, 
        mv: Move
    ) -> Result<(), String>;

    fn best_move(
        &mut self, 
        ctrl: SearchControl,
        on_progress: &mut dyn FnMut(SearchProgress),
    ) -> SearchResult;
    
    fn set_hash_size_mb(
        &mut self, 
        mb: u32,
    );

    fn search_handle(
        &self
    ) -> SearchHandle;
}
