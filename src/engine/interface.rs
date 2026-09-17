use shakmaty::{Move};


struct Score {
    cp: f32, // Score in centipawns
    mate_in: i32, // mate in x moves, negative if getting mated
}


struct SearchInfo {
    depth: u32, // depth in plies
    search_time: u32, // time searched in ms
    nodes_searched: u32,
    pv: Vec<Move>,
    curr_v: Option<Vec<Move>>,
    score: Score,
    curr_move: Move,
    curr_move_num: u32, // the currently searched move, 1 for first move.
    hashfull: u32, // hash full in permill (0 - 1000)
    nodes_per_s: u32,
    gen_string: Option<String>,
}


trait ChessEngine {

}
