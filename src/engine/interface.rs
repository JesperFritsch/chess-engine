use shakmaty::{Move, Chess};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};


pub enum Score {
    Cp(i32), // Score in centipawns
    Mate(i32), // mate in x moves, negative if getting mated
}

#[derive(Default)]
pub enum TimeMode {
    #[default]
    Unbound,
    Fixed(Duration),
    Clock(Clock)
}


pub struct Clock {
    pub remaining: Duration,
    pub opp_remaining: Duration,
    pub increment: Duration,
    pub moves_to_go: Option<u32>,
}

#[derive(Default)]
pub struct Limits {
    pub time_mode: TimeMode,
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


pub struct SearchHandleInner {
    stop: AtomicBool,
    limits: Mutex<Limits>,
}


#[derive(Clone)]
pub struct SearchHandle(
    Arc<SearchHandleInner>
);

impl SearchHandle {
    pub fn new(limits: Limits) -> Self{
        SearchHandle (
            Arc::new(SearchHandleInner {
                stop: AtomicBool::new(false),
                limits: Mutex::new(limits),
            })
        )
    }
    pub fn stop(&self) {self.0.stop.store(true, Ordering::Relaxed)}
    pub fn reset(&self) {self.0.stop.store(false, Ordering::Relaxed)}
    pub fn is_stopped(&self) -> bool {self.0.stop.load(Ordering::Relaxed)}
    pub fn set_limits(&self, limits: Limits) {
        *self.0.limits.lock().unwrap() = limits;
    }
    pub fn with_limits<T>(self, f: impl FnOnce(&Limits) -> T) -> T {
        f(&self.0.limits.lock().unwrap())
    }
}

impl Default for SearchHandle {
    fn default() -> Self { Self::new(Limits::default())}
}


#[derive(Debug)]
pub struct IllegalMove(pub Move);


pub trait ChessEngine {
    fn set_position(
        &mut self, 
        pos: Chess
    );

    fn play_move(
        &mut self, 
        mv: Move
    ) -> Result<(), IllegalMove>;

    fn best_move(
        &mut self, 
        on_progress: &mut dyn FnMut(&SearchProgress),
    ) -> SearchResult;
    
    fn set_hash_size_mb(
        &mut self, 
        mb: usize,
    );

    fn search_handle(
        &self
    ) -> SearchHandle;
}
