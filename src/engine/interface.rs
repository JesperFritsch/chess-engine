use shakmaty::{Move};


pub struct Score {
    cp: f32, // Score in centipawns
    mate_in: i32, // mate in x moves, negative if getting mated
}


pub struct SearchInfo {
    depth: u32, // depth in plies
    search_time: u32, // time searched in ms
    nodes_searched: u32,
    pv: Vec<Move>,
    curr_v: Vec<Move>,
    score: Score,
    curr_move: Move,
    curr_move_num: u32, // the currently searched move, 1 for first move.
    hashfull: u32, // hash full in permill (0 - 1000)
    nodes_per_s: u32,
}


pub struct Options {
    hash_size_mb: u32, // MB size of the hash table
    ponder: bool,
}


pub trait ChessEngine {
    pub fn get_search_info() -> SearchInfo;
    pub fn get_options() -> Options;
    pub fn set_options(options: Options);
}
