use shakmaty::{Chess, Position, Move, Role};
use super::eval::{evaluate, piece_value};

use std::{time::Instant};

const MATE: i32 = 1_000_000;
const MAX_POSSIBLE_SCORE: i32 =
        piece_value(Role::Pawn) * 8
        + piece_value(Role::Knight) * 2
        + piece_value(Role::Bishop) * 2
        + piece_value(Role::Rook) * 2
        + piece_value(Role::Queen) * 1;

pub struct SearchResult {
    pub score: i32,
    pub line: Vec<Move>,
}


fn negamax(
    pos: &Chess, 
    depth: u32, 
    mut alpha: i32,
    beta: i32,
    node_count: &mut u64,
    ply: u32,
    deadline: Instant,
) -> Option<SearchResult> {
    *node_count += 1;
    
    let mut best_score = -(MATE + 1);
    let mut best_line: Vec<Move> = Vec::new();

    if pos.is_game_over() {
        return Some(SearchResult { score: leaf_score(pos, depth), line: best_line });
    }

    if depth == 0 {
        return quiescence(pos, alpha, beta, node_count, deadline);
    }

    if *node_count & 0x7FF == 0 && Instant::now() >= deadline {
        return None; // Time limit reached, return None to indicate search should stop
    }
    
    for m in ordered_moves(pos) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let mut child_result = negamax(&child, depth - 1, -beta, -alpha, node_count, ply + 1, deadline)?;
        let score = -child_result.score;
        if score > best_score {
            best_score = score;
            child_result.line.push(m);
            best_line = child_result.line;
        }

        if score > alpha {
            alpha = score;
        }

        if alpha >= beta {
            break; // Beta cutoff
        }
    }
    Some(SearchResult { score: best_score, line: best_line }) 
}

// make sure that the search does not stop in the middle if a trade or sequence of captures.
fn quiescence(pos: &Chess, mut alpha: i32, beta: i32, node_count: &mut u64, deadline: Instant) -> Option<SearchResult> {
    *node_count += 1;
    
    let stand_pat = evaluate(pos);
    if stand_pat >= beta {
        return Some(SearchResult { score: beta, line: Vec::new() });
    }
    if stand_pat > alpha {
        alpha = stand_pat;        // static eval is our baseline
    }
    let mut best_score = stand_pat;
    let mut best_line: Vec<Move> = Vec::new();

    for m in ordered_moves(pos) {
        if !m.is_conversion() {
            continue; 
        }
        let mut child = pos.clone();
        child.play_unchecked(m);
        let mut child_result = quiescence(&child, -beta, -alpha, node_count, deadline)?;
        let score = -child_result.score;
        if score > best_score {
            best_score = score;
            child_result.line.push(m);
            best_line = child_result.line;
        }

        if score > alpha {
            alpha = score;
        }

        if alpha >= beta {
            break; // Beta cutoff
        }
    }
    Some(SearchResult { score: best_score, line: best_line })
}


fn ordered_moves(pos: &Chess) -> Vec<Move> {
    let mut moves: Vec<Move> = pos.legal_moves().into_iter().collect();
    moves.sort_by_cached_key(|m| {
        let mut score = match m.capture() {
            Some(victim) => 10 * piece_value(victim) - piece_value(m.role()),
            None => 0,
        };
        if let Some(promo) = m.promotion() {
            score += 10 * piece_value(promo);
        }
        std::cmp::Reverse(score)
    });
    moves
}


fn leaf_score(pos: &Chess, ply: u32) -> i32 {
    match pos.outcome().known() {
        Some(outcome) => {
            match outcome.winner() {
                Some(winner) => {
                    let mate = MATE - ply as i32; // Prefer faster mates
                    if winner == pos.turn() { mate } else { -mate }
                }
                None => 0, // Draw
            }
        }
        None => evaluate(pos),
    }
}

pub fn best_line(pos: &Chess, depth: u32, deadline: Instant) -> Option<SearchResult> {
    let mut best_score = -(MATE + 1);
    let mut calc_line: Vec<Move> = Vec::new();
    let mut node_count = 0;
    let mut alpha = -(MATE + 1);
    let beta = MATE + 1;
    for m in ordered_moves(pos) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let child_result = negamax(&child, depth - 1, -beta, -alpha, &mut node_count, 0, deadline)?;
        let score = -child_result.score;
        if score > best_score {
            best_score = score;
            alpha = score;
            calc_line = child_result.line;
            calc_line.push(m);
        }
    }
    Some(SearchResult { score: best_score, line: calc_line.into_iter().rev().collect() })
}

pub fn time_bound_best_line(pos: &Chess, time_limit_ms: u64) -> Option<SearchResult> {
    let deadline = Instant::now() + std::time::Duration::from_millis(time_limit_ms);
    let mut best_result: Option<SearchResult> = None;
    for depth in 1.. {
        if Instant::now() >= deadline {
            break;
        }
        let result = best_line(pos, depth, deadline);
        if let Some(r) = result {
            let is_mate = r.score.abs() >= MAX_POSSIBLE_SCORE;
            best_result = Some(r);
            if is_mate {
                break; // Stop searching deeper if a mate is found
            }
        }
    }
    best_result
}