use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, EnPassantMode, Move, Position, Role};


use super::eval::{evaluate, piece_value};
use super::tt::{Tt, Bound};
use super::interface::{ChessEngine, 
    SearchHandle,
    SearchProgress,
    SearchResult,
    IllegalMove,
    Limits,
    TimeMode,
    Score
};
use std::{time::Instant, time::Duration};

const MATE: i32 = 1_000_000;
const MATE_THRESHOLD: i32 =
    piece_value(Role::Pawn) * 8
    + piece_value(Role::Knight) * 2
    + piece_value(Role::Bishop) * 2
    + piece_value(Role::Rook) * 2
    + piece_value(Role::Queen);

const CHECK_STOP_AFTER: u64 = 2000;

// The deepest ply we keep per-ply killer slots for. Searches never get near
// this in practice; deeper plies just fall back to the last slot.
const MAX_PLY: usize = 128;

pub struct SearchContext {
    pub depth: u8,
    pub node_count: u64,
    pub pv: Vec<Move>,
    pub limits: Limits,
    pub limits_seq: u64,
    pub deadline: Option<Instant>,
    node_check_count: u64,
    // Two "killer" quiet moves per ply: quiet moves that recently caused a beta
    // cutoff at this distance from the root. They tend to work again in sibling
    // positions, so we try them right after captures.
    killers: [[Option<Move>; 2]; MAX_PLY],
    // history[from][to]: how often a quiet move from→to has caused a cutoff,
    // anywhere in the tree. A soft, global ordering signal for quiet moves.
    history: [[i32; 64]; 64],
}

impl Default for SearchContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchContext {
    pub fn new() -> Self {
        SearchContext {
            depth: 0,
            node_count: 0,
            pv: Vec::new(),
            limits: Limits::default(),
            limits_seq: 0,
            deadline: None,
            node_check_count: 0,
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 64],
        }
    }
}


/// The project's search engine, adapted to [`ChessEngine`].
pub struct SearchEngine {
    tt: Tt,
    pos: Chess,
    search_handle: SearchHandle,
    search_ctx: SearchContext
}

impl SearchEngine {
    /// `tt_mb`: transposition-table size in MiB.
    pub fn new(tt_mb: usize) -> Self {
        SearchEngine { 
            tt: Tt::new(tt_mb),
            pos: Chess::default(),
            search_handle: SearchHandle::default(), 
            search_ctx: SearchContext::new()
        }
    }
    

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &Chess,
        mut depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: u8,
        pv: &mut Vec<Move>,
        // Whether null-move pruning is permitted at this node. Disabled inside the
        // null search (no two null moves in a row) and inside the verification
        // search (which must make real moves).
        allow_null: bool,
    ) -> Option<i32> {
        self.search_ctx.node_count += 1;
        pv.clear(); // no continuation until a move raises alpha

        let mut best_score = -(MATE + 1);
        let mut best_move: Option<Move> = None;
        let alpha_orig = alpha;
        if pos.is_game_over() {
            return Some(leaf_score(pos, ply));
        }

        // Check extension: while in check the position is forcing and tactically
        // unresolved, so we must not evaluate it or drop into quiescence here —
        // search one ply deeper instead. The MAX_PLY cap stops an endless checking
        // sequence from recursing forever (we have no repetition detection yet).
        let node_in_check = pos.is_check();
        if node_in_check && (ply as usize) < MAX_PLY {
            depth += 1;
        }

        if depth == 0 {
            return self.quiescence(pos, alpha, beta);
        }

        if self.search_ctx.node_count >= self.search_ctx.node_check_count && self.should_stop() {
            self.search_ctx.node_check_count = self.search_ctx.node_count + CHECK_STOP_AFTER;
            return None; // Time limit reached, return None to indicate search should stop
        }
        
        let key = pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0;

        let mut hash_move: Option<Move> = None;
        if let Some(e) = self.tt.probe(key) {
            // Remember the move this position was best with last time; even when the
            // stored score isn't usable for a cutoff, the move is a great first
            // guess and belongs at the front of the move list.
            hash_move = e.best_move;

            // Never take a TT score cutoff at the root: returning here skips the
            // loop that fills `pv`, which at ply 0 would leave us with an empty PV
            // and therefore no move to play. At the root we always search so a real
            // principal variation is produced.
            if ply != 0 && e.depth >= depth {
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
        // Null-move pruning: hand the opponent a free move (swap the side to move).
        // If our position is still good enough to beat beta even after that — proven
        // by a reduced-depth search — the real position is almost certainly a cutoff,
        // so we prune without searching our own moves. Skipped when:
        //  * we're in check (a "pass" is illegal and the position is forcing),
        //  * beta is a mate score (a null move can't prove a mate), and
        //  * the side to move has only pawns/king (zugzwang: passing isn't "free").
        if allow_null
            && depth >= 3
            && !node_in_check
            && beta.abs() < MATE_THRESHOLD
            && has_non_pawn_material(pos)
        {
            let reduction = 2 + (depth >= 6) as u8;
            if let Ok(null_pos) = pos.clone().swap_turn() {
                let mut null_pv = Vec::new();
                // The null child cannot null again (allow_null = false).
                let null_score = -self.negamax(
                    &null_pos, depth - 1 - reduction, -beta, -beta + 1, ply + 1, &mut null_pv, false,
                )?;
                if null_score >= beta {
                    // Verify the cutoff with a real-move search (null disabled). In
                    // zugzwang the "pass" beats beta but every real move does not, so
                    // this reduced search fails low and we fall through to a normal
                    // search instead of pruning. Otherwise it confirms the cutoff.
                    let mut verify_pv = Vec::new();
                    let verify = self.negamax(
                        pos, depth - reduction, beta - 1, beta, ply, &mut verify_pv, false,
                    )?;
                    if verify >= beta {
                        return Some(beta);
                    }
                }
            }
        }

        let kply = (ply as usize).min(MAX_PLY - 1);
        let mut child_pv: Vec<Move> = Vec::new();
        for (move_count, m) in ordered_moves(pos, hash_move, self.search_ctx.killers[kply], &self.search_ctx.history)
            .into_iter()
            .enumerate()
        {
            let mut child = pos.clone();
            child.play_unchecked(m);

            let score = if move_count == 0 {
                // Principal variation move: full window, full depth. This is the
                // move we believe is best, so we spend the most on it.
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, &mut child_pv, true)?
            } else {
                // Late Move Reductions: given good ordering, quiet moves this far
                // down the list are unlikely to be best, so search them shallower
                // first. Never reduce at shallow depth, the first few moves,
                // tactical moves, or moves that give/escape check.
                let reduce = depth >= 3
                    && move_count >= 3
                    && is_quiet(m)
                    && !node_in_check
                    && !child.is_check();
                let reduction = if reduce { lmr_reduction(depth, move_count) } else { 0 };

                // Reduced-depth zero-window "scout": just asks "does this beat alpha?".
                let mut s = -self.negamax(
                    &child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, &mut child_pv, true,
                )?;

                // A reduced move that beat alpha wasn't as weak as we assumed:
                // re-search it at full depth (still zero-window) for an honest verdict.
                if reduction > 0 && s > alpha {
                    s = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, ply + 1, &mut child_pv, true)?;
                }

                // PVS: any move that beats alpha inside the window may be a new
                // principal variation, so re-search it with the full window for an
                // exact score and PV.
                if s > alpha && s < beta {
                    s = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, &mut child_pv, true)?;
                }
                s
            };

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
                // Beta cutoff. If the move that caused it was *quiet* (not a
                // capture/promotion — those are already ordered well by MVV-LVA),
                // reward it so we try it earlier in sibling and future positions.
                if is_quiet(m) {
                    record_killer(&mut self.search_ctx, kply, m);
                    if let Some(from) = m.from() {
                        // Deeper cutoffs are worth more, hence depth² (a common weight).
                        self.search_ctx.history[from.to_usize()][m.to().to_usize()] +=
                            (depth * depth) as i32;
                    }
                }
                break;
            }
        }

        let bound = if best_score <= alpha_orig {
            Bound::Upper          // never raised alpha → upper bound
        } else if best_score >= beta {
            Bound::Lower          // beta cutoff → lower bound
        } else {
            Bound::Exact          // alpha < score < beta → exact
        };
        self.tt.store(key, depth, score_to_tt(best_score, ply), bound, best_move);

        Some(best_score) 
    } 

    fn quiescence(
        &mut self,
        pos: &Chess, 
        mut alpha: i32, 
        beta: i32, 
    ) -> Option<i32> {
        self.search_ctx.node_count += 1;
        
        let stand_pat = evaluate(pos);
        if stand_pat >= beta {
            return Some(stand_pat);
        }
        if stand_pat > alpha {
            alpha = stand_pat;        // static eval is our baseline
        }
        let mut best_score = stand_pat;

        // Quiescence only looks at captures/promotions, so killer/history ordering
        // is irrelevant here — pass empty slots.
        for m in ordered_moves(pos, None, [None, None], &self.search_ctx.history) {
            if !m.is_conversion() {
                continue;
            }
            let mut child = pos.clone();
            child.play_unchecked(m);
            let score = -self.quiescence(&child, -beta, -alpha)?;
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

    pub fn depth_bound_search(
        &mut self,
        pos: &Chess,
        depth: u8
    ) -> Option<i32> {
        let mut pv = Vec::new();
        let result = self.negamax(pos, depth, -(MATE + 1), MATE + 1, 0, &mut pv, true);
        if result.is_some() {
            // Only commit the PV of a fully completed search (not a timed-out one).
            self.search_ctx.pv = pv;
        }
        result
    } 

    fn should_stop(&mut self) -> bool {
        let seq = self.search_handle.limits_seq();
        let now = Instant::now();
        if seq != self.search_ctx.limits_seq {
            self.search_ctx.limits = self.search_handle.with_limits(|l| l.clone());
            self.search_ctx.deadline = calc_deadline(&self.search_ctx.limits, now);
            self.search_ctx.limits_seq = seq;
        }
        if self.search_ctx.deadline.is_some_and(|i| now >= i)
            || self.search_ctx.limits.max_nodes.is_some_and(|n| self.search_ctx.node_count >= n)
            || self.search_handle.is_stopped() {
            return true;
        }
        false
    }

    fn reset_ctx(&mut self) {
        self.search_ctx.limits = self.search_handle.with_limits(|l| l.clone());
        self.search_ctx.limits_seq = self.search_handle.limits_seq();
        self.search_ctx.deadline = calc_deadline(&self.search_ctx.limits, Instant::now());
        self.search_ctx.depth = 0;
        self.search_ctx.node_check_count = 0;
        self.search_ctx.node_count = 0;
        self.search_ctx.pv.clear();
        self.search_ctx.killers = [[None; 2]; MAX_PLY];
    }

}

impl ChessEngine for SearchEngine {
    fn set_position(
        &mut self, 
        pos: Chess
    ) {
        self.pos = pos
    }

    fn play_move(
        &mut self, 
        mv: Move
    ) -> Result<(), IllegalMove> {
        if !self.pos.is_legal(mv) {
            return Err(IllegalMove(mv));
        }
        self.pos.play_unchecked(mv);
        Ok(())
    }
    
    fn search(
        &mut self,
        on_progress: &mut dyn FnMut(&SearchProgress),
    ) -> SearchResult {
        let start = Instant::now();
        let mut best_result: Option<i32> = None;
        let pos = self.pos.clone();
        self.reset_ctx();
        for depth in 1.. {
            if self.search_ctx.limits.max_depth.is_some_and(|d| depth > d) { break }
            let result = self.depth_bound_search(&pos, depth);
            let Some(r) = result else { break };
            self.search_ctx.depth = depth;
            best_result = result;
            on_progress(&SearchProgress {
                depth,
                seldepth: depth,
                score: mate_in(r).map(Score::Mate).unwrap_or(Score::Cp(r)),
                nodes: self.search_ctx.node_count,
                pv: &self.search_ctx.pv,
                elapsed: start.elapsed(),
                nodes_per_s: 0,
                hashfull: (self.tt.fill_fraction() * 1000.0) as u32

            });
            if r.abs() >= MATE_THRESHOLD && !matches!(self.search_ctx.limits.time_mode, TimeMode::Unbound) {
                break; // Stop searching deeper if a mate is found
            }
        }
        SearchResult {
            best_move: self.search_ctx.pv.first().copied(),
            pv: self.search_ctx.pv.clone(),
            score: best_result.map(to_score),
            depth: self.search_ctx.depth,
            nodes: self.search_ctx.node_count
        }
    }

    fn set_hash_size_mb(
        &mut self, 
        mb: usize,
    ) {
        self.tt.resize(mb);
    }

    fn clear(
        &mut self
    ) {
        self.tt.clear();
    }

    fn search_handle(
        &self
    ) -> SearchHandle {
        self.search_handle.clone()
    }

}

// store: convert node-relative → mate-relative (absolute)
fn score_to_tt(score: i32, ply: u8) -> i32 {
    if score >= MATE_THRESHOLD { score + ply as i32 }
    else if score <= -MATE_THRESHOLD { score - ply as i32 }
    else { score }
}


// probe: convert mate-relative → node-relative
fn score_from_tt(score: i32, ply: u8) -> i32 {
    if score >= MATE_THRESHOLD { score - ply as i32 }
    else if score <= -MATE_THRESHOLD { score + ply as i32 }
    else { score }
}


fn calc_deadline(limits: &Limits, time_now: Instant) -> Option<Instant> {
    match limits.time_mode {
        TimeMode::Unbound => None,
        TimeMode::Fixed(duration) => {
            Some(time_now + duration)
        },
        TimeMode::Clock(clock) => {
            let base = match clock.moves_to_go {
                Some(m) => clock.remaining / m.max(1) + clock.increment * 3 / 4,
                None => clock.remaining / 25 + clock.increment * 3 / 4,
            };
            let cap = clock.remaining.saturating_sub(Duration::from_millis(100)) / 2;
            Some(time_now + base.min(cap))
        }
    }
}


// Move-ordering score tiers, from most to least promising. The gaps keep the
// tiers from overlapping: captures/promotions always outrank killers, which
// always outrank history-ordered quiets.
const HASH_MOVE_SCORE: i32 = 1_000_000;
const CAPTURE_BASE: i32 = 100_000; // + MVV-LVA (≈ up to 9000)
const KILLER_1_SCORE: i32 = 90_000;
const KILLER_2_SCORE: i32 = 80_000;
const HISTORY_MAX: i32 = 70_000; // cap quiet history so it can't reach the killer tier

fn ordered_moves(
    pos: &Chess,
    hash_move: Option<Move>,
    killers: [Option<Move>; 2],
    history: &[[i32; 64]; 64],
) -> Vec<Move> {
    let mut moves: Vec<Move> = pos.legal_moves().into_iter().collect();
    moves.sort_by_cached_key(|m| std::cmp::Reverse(move_order_score(*m, hash_move, killers, history)));
    moves
}

fn move_order_score(
    m: Move,
    hash_move: Option<Move>,
    killers: [Option<Move>; 2],
    history: &[[i32; 64]; 64],
) -> i32 {
    if Some(m) == hash_move {
        return HASH_MOVE_SCORE;
    }

    // Captures and promotions form the "tactical" tier.
    let mut tactical = 0;
    if let Some(victim) = m.capture() {
        // MVV-LVA: most valuable victim, least valuable attacker.
        tactical = CAPTURE_BASE + 10 * piece_value(victim) - piece_value(m.role());
    }
    if let Some(promo) = m.promotion() {
        tactical = tactical.max(CAPTURE_BASE) + 10 * piece_value(promo);
    }
    if tactical != 0 {
        return tactical;
    }

    // Quiet moves: killers first, then the history table orders the rest.
    if Some(m) == killers[0] {
        return KILLER_1_SCORE;
    }
    if Some(m) == killers[1] {
        return KILLER_2_SCORE;
    }
    match m.from() {
        Some(from) => history[from.to_usize()][m.to().to_usize()].min(HISTORY_MAX),
        None => 0,
    }
}

fn is_quiet(m: Move) -> bool {
    m.capture().is_none() && m.promotion().is_none()
}

/// Whether the side to move has any piece beyond pawns and the king. Used to
/// disable null-move pruning in likely-zugzwang (pawn-only) positions.
fn has_non_pawn_material(pos: &Chess) -> bool {
    let turn = pos.turn();
    pos.board()
        .clone()
        .into_iter()
        .any(|(_, p)| p.color == turn && p.role != Role::Pawn && p.role != Role::King)
}

/// How many plies to shave off a late quiet move. Grows slowly (logarithmically)
/// with both remaining depth and how late the move is, and is clamped so the
/// reduced search still has at least one ply left.
fn lmr_reduction(depth: u8, move_count: usize) -> u8 {
    let r = 0.75 + (depth as f64).ln() * (move_count as f64).ln() / 2.25;
    (r as u8).clamp(1, depth.saturating_sub(2))
}

fn record_killer(ctx: &mut SearchContext, ply: usize, m: Move) {
    // Keep two distinct killers, most-recent first.
    if ctx.killers[ply][0] != Some(m) {
        ctx.killers[ply][1] = ctx.killers[ply][0];
        ctx.killers[ply][0] = Some(m);
    }
}


fn leaf_score(pos: &Chess, ply: u8) -> i32 {
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


/// If `score` is a forced-mate score, return the mate distance in moves —
/// positive if the side to move is delivering mate, negative if being mated.
pub fn mate_in(score: i32) -> Option<i32> {
    if score >= MATE_THRESHOLD {
        Some((MATE - score + 1) / 2)
    } else if score <= -MATE_THRESHOLD {
        Some(-((MATE + score + 1) / 2))
    } else {
        None
    }
}

pub fn to_score(score: i32) -> Score {
    mate_in(score).map(Score::Mate).unwrap_or(Score::Cp(score))
}

