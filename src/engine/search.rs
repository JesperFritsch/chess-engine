use shakmaty::{Chess, Position, Move};
use super::eval::evaluate;

const MATE: i32 = 1_000_000;


pub struct SearchResult {
    pub score: i32,
    pub line: Vec<Move>,
}


fn negamax(pos: &Chess, depth: u32, node_count: &mut u64) -> (i32, Vec<Move>) {
    *node_count += 1;
    
    let mut best_score = -(MATE + 1);
    let mut best_line: Vec<Move> = Vec::new();

    if pos.is_game_over() || depth == 0 {
        return (leaf_score(pos, depth), best_line);
    }
    
    for m in pos.legal_moves() {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let (child_score, mut child_line) = negamax(&child, depth - 1, node_count);
        let score = -child_score;
        if score > best_score {
            best_score = score;
            child_line.push(m);
            best_line = child_line;
        }
    }
    (best_score, best_line) 
}

fn leaf_score(pos: &Chess, _depth: u32) -> i32 {
    match pos.outcome().known() {
        Some(outcome) => {
            match outcome.winner() {
                Some(winner) => {
                    if winner == pos.turn() { MATE } else { -MATE }
                }
                None => 0, // Draw
            }
        }
        None => evaluate(pos),
    }
}

pub fn best_line(pos: &Chess, depth: u32) -> SearchResult {
    let mut best_score = -(MATE + 1);
    let mut calc_line: Vec<Move> = Vec::new();
    for m in pos.legal_moves() {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let mut node_count = 0;
        let (move_score, move_line) = negamax(&child, depth - 1, &mut node_count);
        let score = -move_score;
        if score > best_score {
            best_score = score;
            calc_line = move_line;
            calc_line.push(m);
        }
    }
    SearchResult { score: best_score, line: calc_line.into_iter().rev().collect() }
}