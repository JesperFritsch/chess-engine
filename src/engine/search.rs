use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, EnPassantMode, Move, Position, Role};

use super::eval::{evaluate, piece_value};
use super::tt::{Tt, Bound};

use std::{time::Instant, time::Duration};

const MATE: i32 = 1_000_000;
const MATE_THRESHOLD: i32 =
    piece_value(Role::Pawn) * 8
    + piece_value(Role::Knight) * 2
    + piece_value(Role::Bishop) * 2
    + piece_value(Role::Rook) * 2
    + piece_value(Role::Queen) * 1;


pub struct SearchContext {
    pub tt: Tt,
    pub node_count: u64,
    pub deadline: Instant,
    pub pv: Vec<Move>,
}


impl SearchContext {
    pub fn new(tt_mb_size: usize, ms_timeout: u64) -> Self {
        SearchContext {
            tt: Tt::new(tt_mb_size),
            node_count: 0,
            deadline: Instant::now() + Duration::from_millis(ms_timeout),
            pv: Vec::new(),
        }
    }
}

// store: convert node-relative → mate-relative (absolute)
fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE_THRESHOLD { score + ply as i32 }
    else if score <= -MATE_THRESHOLD { score - ply as i32 }
    else { score }
}


// probe: convert mate-relative → node-relative
fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE_THRESHOLD { score - ply as i32 }
    else if score <= -MATE_THRESHOLD { score + ply as i32 }
    else { score }
}


fn negamax(
    search_ctx: &mut SearchContext,
    pos: &Chess,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    pv: &mut Vec<Move>,
) -> Option<i32> {
    search_ctx.node_count += 1;
    pv.clear(); // no continuation until a move raises alpha

    let mut best_score = -(MATE + 1);
    let mut best_move: Option<Move> = None;
    let alpha_orig = alpha;
    if pos.is_game_over() {
        return Some(leaf_score(pos, ply));
    }

    if depth == 0 {
        return quiescence(search_ctx, pos, alpha, beta);
    }

    if search_ctx.node_count & 0x7FF == 0 && Instant::now() >= search_ctx.deadline {
        return None; // Time limit reached, return None to indicate search should stop
    }
    
    let key = pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0;

    if let Some(e) = search_ctx.tt.probe(key) {
        if e.depth >= depth {
            let usable = match e.bound {
                Bound::Exact => true,
                Bound::Lower => e.score >= beta,
                Bound::Upper => e.score <= alpha,
            };
            if usable {
                return Some(score_from_tt(e.score, ply));
            }
        }
    }
    let mut child_pv: Vec<Move> = Vec::new();
    for m in ordered_moves(pos) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let score = -negamax(search_ctx, &child, depth - 1, -beta, -alpha, ply + 1, &mut child_pv)?;
        if score > best_score {
            best_score = score;
            best_move = Some(m);
        }

        if score > alpha {
            alpha = score;
            // New best line at this node: this move followed by the child's PV.
            pv.clear();
            pv.push(m);
            pv.extend_from_slice(&child_pv);
        }

        if alpha >= beta {
            break; // Beta cutoff
        }
    }

    let bound = if best_score <= alpha_orig {
        Bound::Upper          // never raised alpha → upper bound
    } else if best_score >= beta {
        Bound::Lower          // beta cutoff → lower bound
    } else {
        Bound::Exact          // alpha < score < beta → exact
    };
    search_ctx.tt.store(key, depth, score_to_tt(best_score, ply), bound, best_move);

    Some(best_score) 
}

// make sure that the search does not stop in the middle if a trade or sequence of captures.
fn quiescence(
    search_ctx: &mut SearchContext,
    pos: &Chess, 
    mut alpha: i32, 
    beta: i32, 
) -> Option<i32> {
    search_ctx.node_count += 1;
    
    let stand_pat = evaluate(pos);
    if stand_pat >= beta {
        return Some(stand_pat);
    }
    if stand_pat > alpha {
        alpha = stand_pat;        // static eval is our baseline
    }
    let mut best_score = stand_pat;

    for m in ordered_moves(pos) {
        if !m.is_conversion() {
            continue; 
        }
        let mut child = pos.clone();
        child.play_unchecked(m);
        let score = -quiescence(search_ctx, &child, -beta, -alpha)?;
        if score > best_score {
            best_score = score;
        }

        if score > alpha {
            alpha = score;
        }

        if alpha >= beta {
            break; // Beta cutoff
        }
    }
    Some(best_score)
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

pub fn depth_bound_search(search_ctx: &mut SearchContext, pos: &Chess, depth: u32) -> Option<i32> {
    let mut pv = Vec::new();
    let result = negamax(search_ctx, pos, depth, -(MATE + 1), MATE + 1, 0, &mut pv);
    if result.is_some() {
        // Only commit the PV of a fully completed search (not a timed-out one).
        search_ctx.pv = pv;
    }
    result
}

pub fn time_bound_search(search_ctx: &mut SearchContext, pos: &Chess) -> Option<i32> {
    let mut best_result: Option<i32> = None;
    for depth in 1.. {
        if Instant::now() >= search_ctx.deadline {
            break;
        }
        let result = depth_bound_search(search_ctx, pos, depth);
        if let Some(r) = result {
            best_result = result;
            let is_mate = r.abs() >= MATE_THRESHOLD;
            if is_mate {
                break; // Stop searching deeper if a mate is found
            }
        }
    }
    best_result
}