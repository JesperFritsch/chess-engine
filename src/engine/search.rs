use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, EnPassantMode, Move, Position, Role, MoveList};


use super::eval::{evaluate, piece_value};
use super::tt::{Tt, Bound, PackedMove, pack_move};
use super::interface::{ChessEngine, 
    SearchHandle,
    SearchProgress,
    SearchResult,
    IllegalMove,
    Limits,
    TimeMode,
    Score
};
use core::hash;
use::std::sync::atomic::{
    AtomicI32,
    AtomicU64,
    AtomicUsize,
    Ordering
};
use std::{time::Instant, time::Duration};
use arrayvec::ArrayVec;
// Kept inside i16 so that a whole entry fits in one u64 in the
// transposition table.
const MATE: i32 = 32_000;
const MATE_THRESHOLD: i32 =
    piece_value(Role::Pawn) * 8
    + piece_value(Role::Knight) * 2
    + piece_value(Role::Bishop) * 2
    + piece_value(Role::Rook) * 2
    + piece_value(Role::Queen);

// Score of a drawn position. Kept as a name so a contempt term (preferring a
// slightly-worse-than-equal fight over a draw) has one place to live later.
const DRAW: i32 = 0;

const CHECK_STOP_AFTER: u64 = 2000;

// The deepest ply we keep per-ply killer slots for. Searches never get near
// this in practice; deeper plies just fall back to the last slot.
const MAX_PLY: usize = 128;
const MAX_DEPTH: u8 = 255;


struct Shared {
    history: [[AtomicI32; 64]; 64],
    handle: SearchHandle,
    prev_pos_keys: Vec<Zobrist64>,
    base: Instant,
    deadline_nanos: AtomicU64,
    max_nodes: AtomicU64,
    limits_seq: AtomicU64,
    nodes: AtomicU64,
    threads: usize,
    busy:AtomicUsize,
}

impl Shared {
    fn new(sh: SearchHandle) -> Shared {
        Shared {
            history: std::array::from_fn(|_| std::array::from_fn(|_| AtomicI32::new(0))),
            handle: sh,
            prev_pos_keys: Vec::new(),
            base: Instant::now(),
            deadline_nanos: AtomicU64::new(0),
            max_nodes: AtomicU64::new(0),
            limits_seq: AtomicU64::new(0),
            nodes: AtomicU64::new(0),
            threads: 0,
            busy: AtomicUsize::new(0),
        }
    }

    fn begin(&mut self, prev_pos_keys: &Vec<Zobrist64>) {
        self.history = std::array::from_fn(|_| std::array::from_fn(|_| AtomicI32::new(0)));
        self.prev_pos_keys = prev_pos_keys.clone();
        self.base = Instant::now();
        self.update_limits();
        self.nodes.store(0, Ordering::Relaxed);
    }

    fn update_limits(&self) {
        let now = Instant::now();
        self.handle.with_limits(|l| {
            self.max_nodes.store(l.max_nodes.unwrap_or(u64::MAX), Ordering::Relaxed);
            let nanos = if let Some(deadline) = calc_deadline(l, now) {
                deadline.saturating_duration_since(self.base).as_nanos() as u64
            } else {
                u64::MAX
            };
            self.deadline_nanos.store(nanos, Ordering::Relaxed);
        });
        self.limits_seq.store(self.handle.limits_seq(), Ordering::Relaxed);
    }

    fn should_stop(&self) -> bool {
        let seq = self.handle.limits_seq();
        if seq != self.limits_seq.load(Ordering::Relaxed) {
            self.update_limits();
        }
        if self.base.elapsed().as_nanos() as u64 > self.deadline_nanos.load(Ordering::Relaxed)
            || self.nodes.load(Ordering::Relaxed) >= self.max_nodes.load(Ordering::Relaxed)
            || self.handle.is_stopped() {
            return true;
        }
        false
    }
    

}

struct Worker {
    killers: [[Option<Move>; 2]; MAX_PLY],
    pos_keys: Vec<Zobrist64>,
    null_floor: usize,
    nodes: u64,
    nodes_flushed: u64,
    nodes_check_count: u64,
    child_pv: Vec<Move>,
}

impl Worker {
    fn new() -> Worker {
        Worker {
            killers: [[None; 2]; MAX_PLY],
            pos_keys: Vec::new(),
            null_floor: 0,
            nodes: 0,
            nodes_flushed: 0,
            nodes_check_count: 0,
            child_pv: Vec::new(),
        }
    }
}

struct NegaCtx<'a> {
    tt: &'a Tt,
    worker: &'a mut Worker,
    shared: &'a Shared, 
}

/// The project's search engine, adapted to [`ChessEngine`].
pub struct SearchEngine {
    tt: Tt,
    shared: Shared,
    pos: Chess,
    // Keys of every position the game has visited before `pos`, oldest first.
    // Seeded by `set_position` and extended by `play_move`; copied into the
    // search context as the base of the repetition stack at the start of a search.
    pos_history_keys: Vec<Zobrist64>,
    search_handle: SearchHandle,
}

impl SearchEngine {
    /// `tt_mb`: transposition-table size in MiB.
    pub fn new(tt_mb: usize) -> Self {
        let sh = SearchHandle::default();
        SearchEngine { 
            tt: Tt::new(tt_mb),
            pos: Chess::default(),
            pos_history_keys: vec!(),
            shared: Shared::new(sh.clone()),
            search_handle: sh, 
        }
    } 
}


impl ChessEngine for SearchEngine {
    fn set_position(
        &mut self,
        start: Chess,
        moves: &[Move],
    ) {
        self.pos = start;
        self.pos_history_keys.clear();
        // Replaying rather than accepting a finished position is the whole point:
        // it is the intermediate positions that repetition detection compares
        // against. An illegal move can only come from a caller that built the list
        // against a different starting position, so stop there and keep what is
        // consistent instead of guessing.
        for &mv in moves {
            if self.play_move(mv).is_err() {
                break;
            }
        }
    }

    fn play_move(
        &mut self, 
        mv: Move
    ) -> Result<(), IllegalMove> {
        if !self.pos.is_legal(mv) {
            return Err(IllegalMove(mv));
        }
        // The key of the position being *left*. The current position is kept off
        // the stack so that `negamax` can own that entry: it pushes the node it is
        // searching only when it recurses into a child.
        self.pos_history_keys.push(self.pos.zobrist_hash(EnPassantMode::Legal));
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
        let mut w = Worker::new();
        let mut root_pv = Vec::new();
        let mut completed_depths = 0;
        self.shared.begin(&self.pos_history_keys);
        let mut root_moves: MoveList = pos.legal_moves();
        if let Some(r) = self.search_handle.with_limits(|l| l.restrict_to.clone()) {
            root_moves.retain(|m| r.contains(m));
        }
        for depth in 1..=MAX_DEPTH {
            // A finished game never reaches the stop check in `negamax`.
            if pos.is_game_over() { break }
            if self.search_handle.with_limits(|l| l.max_depth.is_some_and(|d| depth > d)) { break }
            
            w.pos_keys.clone_from(&self.shared.prev_pos_keys);
            w.null_floor = 0;

            let mut ctx = NegaCtx {
                worker: &mut w,
                tt:     &self.tt,
                shared: &self.shared,
            };

            let mut pv = Vec::new();
            let Some(result) = search_root(&mut ctx, &pos, depth, root_moves.clone(), &mut pv) else { break };
            flush_nodes(ctx.worker, ctx.shared);
            best_result = Some(result);
            completed_depths = depth;
            root_pv = pv.clone();
            on_progress(&SearchProgress {
                depth,
                seldepth: depth,
                score: mate_in(result).map(Score::Mate).unwrap_or(Score::Cp(result)),
                nodes: self.shared.nodes.load(Ordering::Relaxed),
                pv: &pv,
                elapsed: start.elapsed(),
                nodes_per_s: 0,
                hashfull: (self.tt.fill_fraction() * 1000.0) as u32

            });
            if result.abs() >= MATE_THRESHOLD {
                break; // Stop searching deeper if a mate is found
            }
        }
        // `go infinite`/`go ponder` may not answer before `stop` or `ponderhit`.
        while matches!(self.search_handle.with_limits(|l| l.clone()).time_mode, TimeMode::Infinite) && !self.shared.should_stop() {
            std::thread::sleep(Duration::from_millis(2));
        }
        SearchResult {
            best_move: root_pv.first().copied(),
            pv: root_pv,
            score: best_result.map(to_score),
            depth: completed_depths,
            nodes: self.shared.nodes.load(Ordering::Relaxed),
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
        // A new game shares nothing with the old one, least of all its repetitions.
        self.pos_history_keys.clear();
    }

    fn search_handle(
        &self
    ) -> SearchHandle {
        self.search_handle.clone()
    }

}



#[allow(clippy::too_many_arguments)]
fn negamax(
    ctx: &mut NegaCtx<'_>,
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
    ctx.worker.nodes += 1;
    pv.clear(); // no continuation until a move raises alpha

    let mut best_score = -(MATE + 1);
    let mut best_move: Option<Move> = None;
    let alpha_orig = alpha;
    if pos.is_game_over() {
        return Some(leaf_score(pos, ply));
    }

    // Wanted by the repetition check just below and by the TT probe further
    // down, so it is computed once here. The hash covers the position but not
    // the halfmove clock or move number, which is exactly what repetition
    // detection needs: the same layout reached twice must compare equal.
    let key = pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);

    // Draws by repetition and by the fifty-move rule. Both are properties of
    // the *path* to this position rather than of the position itself, which is
    // why they are settled before the transposition table is consulted and why
    // the resulting score is never stored in it.
    //
    // Not at the root: the root is by definition already on its own path, and
    // returning here would skip the move loop and leave `pv` empty — the same
    // trap the TT cutoff below avoids at ply 0. Checkmate outranks both rules
    // and has already been handled by `is_game_over` above.
    if ply != 0
        && (pos.halfmoves() >= 100
            || is_repetition(
                &ctx.worker.pos_keys,
                key,
                pos.halfmoves(),
                ctx.worker.null_floor,
            ))
    {
        return Some(DRAW);
    }

    // Check extension: while in check the position is forcing and tactically
    // unresolved, so we must not evaluate it or drop into quiescence here —
    // search one ply deeper instead. The MAX_PLY cap bounds the recursion; an
    // endless checking sequence is now also caught by the repetition check
    // above, which scores a perpetual as the draw it is.
    let node_in_check = pos.is_check();
    if node_in_check && (ply as usize) < MAX_PLY {
        depth += 1;
    }

    if depth == 0 {
        return quiescence(ctx, pos, alpha, beta);
    }

    if ctx.worker.nodes >= ctx.worker.nodes_check_count {
        ctx.worker.nodes_check_count = ctx.worker.nodes + CHECK_STOP_AFTER;
        flush_nodes(ctx.worker, ctx.shared);
        if depth > 0 && ctx.shared.should_stop() {
            return None; 
        }
    } 
    
    let mut hash_move: PackedMove = 0;
    if let Some(e) = ctx.tt.probe(key.0) {
        // Remember the move this position was best with last time; even when the
        // stored score isn't usable for a cutoff, the move is a great first
        // guess and belongs at the front of the move list.
        hash_move = e.mv;

        // Never take a TT score cutoff at the root: returning here skips the
        // loop that fills `pv`, which at ply 0 would leave us with an empty PV
        // and therefore no move to play. At the root we always search so a real
        // principal variation is produced.
        if ply != 0 && e.depth >= depth {
            let score = e.score as i32;
            let usable = match e.bound {
                Bound::Exact => true,
                Bound::Lower => score >= beta,
                Bound::Upper => score <= alpha,
            };
            if usable {
                return Some(score_from_tt(score, ply));
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

            // Bar the null subtree from matching positions earlier on the path.
            // It is reached through a pass, which is not a legal move, so a line
            // in there that "repeats" a real game position does not actually
            // repeat anything. `swap_turn` carries the halfmove clock across
            // unchanged, so the clock alone gives no hint that a pass happened.
            let saved_floor = ctx.worker.null_floor;
            ctx.worker.null_floor = ctx.worker.pos_keys.len();

            // The null child cannot null again (allow_null = false).
            let null_result = negamax(
                ctx, &null_pos, depth - 1 - reduction, -beta, -beta + 1, ply + 1, &mut null_pv, false,
            );

            // Restored before `?` so a search that times out inside the null
            // subtree cannot leave the barrier raised for its caller.
            ctx.worker.null_floor = saved_floor;
            let null_score = -null_result?;
            if null_score >= beta {
                // Verify the cutoff with a real-move search (null disabled). In
                // zugzwang the "pass" beats beta but every real move does not, so
                // this reduced search fails low and we fall through to a normal
                // search instead of pruning. Otherwise it confirms the cutoff.
                let mut verify_pv = Vec::new();
                let verify = negamax(
                    ctx, pos, depth - reduction, beta - 1, beta, ply, &mut verify_pv, false,
                )?;
                if verify >= beta {
                    return Some(beta);
                }
            }
        }
    }

    let kply = (ply as usize).min(MAX_PLY - 1);
    let mut child_pv: Vec<Move> = Vec::new();
    let mut moves = pos.legal_moves();
    ctx.worker.pos_keys.push(key);
    let killers = ctx.worker.killers[kply];
    for (move_count, m) in OrderedMoves::new(pos, hash_move, killers, &ctx.shared.history).enumerate() {
        let score = search_move(ctx, pos, move_count, m, depth, alpha, beta, ply, node_in_check, &mut child_pv)?;

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
                record_killer(&mut ctx.worker, kply, m);
                if let Some(from) = m.from() {
                    // Deeper cutoffs are worth more, hence depth² (a common weight).
                    ctx.shared.history[from.to_usize()][m.to().to_usize()].fetch_add((depth * depth) as i32, Ordering::Relaxed);
                }
            }
            break;
        }
    }

    ctx.worker.pos_keys.pop();

    let bound = if best_score <= alpha_orig {
        Bound::Upper          // never raised alpha → upper bound
    } else if best_score >= beta {
        Bound::Lower          // beta cutoff → lower bound
    } else {
        Bound::Exact          // alpha < score < beta → exact
    };
    ctx.tt.store(key.0, depth, score_to_tt(best_score, ply) as i16, bound, packed(best_move));

    Some(best_score) 
} 



fn quiescence(
    ctx: &mut NegaCtx,
    pos: &Chess, 
    mut alpha: i32, 
    beta: i32, 
) -> Option<i32> {
    ctx.worker.nodes += 1;
    
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
    for (move_count, m) in OrderedMoves::new(pos, 0, [None, None], &ctx.shared.history).filter(|m| m.is_conversion()).enumerate() {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let score = -quiescence(ctx, &child, -beta, -alpha)?;
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

fn search_move(
    ctx: &mut NegaCtx<'_>,
    pos: &Chess,
    move_count: usize,
    mv: Move,
    depth: u8,
    alpha: i32,
    beta: i32,
    ply: u8,
    node_in_check: bool,
    child_pv: &mut Vec<Move>
) -> Option<i32> {

    let mut child = pos.clone();
    child.play_unchecked(mv);

    if move_count == 0 {
        // Principal variation move: full window, full depth. This is the
        // move we believe is best, so we spend the most on it.
        Some(-negamax(ctx, &child, depth - 1, -beta, -alpha, ply + 1, child_pv, true)?)
    } else {
        // Late Move Reductions: given good ordering, quiet moves this far
        // down the list are unlikely to be best, so search them shallower
        // first. Never reduce at shallow depth, the first few moves,
        // tactical moves, or moves that give/escape check.
        let reduce = depth >= 3
            && move_count >= 3
            && is_quiet(mv)
            && !node_in_check
            && !child.is_check();
        let reduction = if reduce { lmr_reduction(depth, move_count) } else { 0 };

        // Reduced-depth zero-window "scout": just asks "does this beat alpha?".
        let mut s = -negamax(
            ctx, &child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, child_pv, true,
        )?;

        // A reduced move that beat alpha wasn't as weak as we assumed:
        // re-search it at full depth (still zero-window) for an honest verdict.
        if reduction > 0 && s > alpha {
            s = -negamax(ctx, &child, depth - 1, -alpha - 1, -alpha, ply + 1, child_pv, true)?;
        }

        // PVS: any move that beats alpha inside the window may be a new
        // principal variation, so re-search it with the full window for an
        // exact score and PV.
        if s > alpha && s < beta {
            s = -negamax(ctx, &child, depth - 1, -beta, -alpha, ply + 1, child_pv, true)?;
        }
        Some(s)
    }
}

fn search_root(
    ctx: &mut NegaCtx,
    pos: &Chess,
    mut depth: u8,
    root_moves: MoveList,
    pv: &mut Vec<Move>,
) -> Option<i32> {
    ctx.worker.nodes += 1;
    pv.clear();
    if pos.is_game_over() { return Some(leaf_score(pos, 0)) };
    let key = pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
    if pos.is_check() { depth += 1; }
    let hash_move = ctx.tt.probe(key.0).map_or(0, |e| e.mv);
    let mut alpha = -(MATE + 1);
    let beta = MATE + 1;
    let mut best_score = -(MATE + 1);
    let mut best_move = None;
    let node_in_check = pos.is_check();
    ctx.worker.pos_keys.push(key);
    
    let mut child_pv: Vec<Move> = Vec::new();

    for (move_count, m) in OrderedMoves::from_list(root_moves, hash_move, ctx.worker.killers[0], &ctx.shared.history).enumerate() {
        let score = search_move(ctx, pos, move_count, m, depth, alpha, beta, 0, node_in_check, &mut child_pv)?;

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
                record_killer(&mut ctx.worker, 0, m);
                if let Some(from) = m.from() {
                    // Deeper cutoffs are worth more, hence depth² (a common weight).
                    ctx.shared.history[from.to_usize()][m.to().to_usize()].fetch_add((depth * depth) as i32, Ordering::Relaxed);
                }
            }
            break;
        }
    }
    ctx.worker.pos_keys.pop();
    ctx.tt.store(key.0, depth, score_to_tt(best_score, 0) as i16, Bound::Exact, packed(best_move));
    Some(best_score)
}

fn flush_nodes(w: &mut Worker, s: &Shared) {
    s.nodes.fetch_add(w.nodes - w.nodes_flushed, Ordering::Relaxed);
    w.nodes_flushed = w.nodes;
}

fn packed(m: Option<Move>) -> PackedMove {
    m.map_or(0, pack_move)
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
        TimeMode::Infinite => None,
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


fn move_order_score(
    m: Move,
    hash_move: PackedMove,
    killers: [Option<Move>; 2],
    history: &[[AtomicI32; 64]; 64],
) -> i32 {
    // The table stores a lossy 16-bit move, so the hash move is found by packing
    // each candidate instead of comparing `Move`s. A packed value that matches
    // nothing here — a wrong hit, or a move that is no longer legal — just fails
    // to match, which is exactly the validation we want.
    if hash_move != 0 && pack_move(m) == hash_move {
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
        Some(from) => history[from.to_usize()][m.to().to_usize()].load(Ordering::Relaxed).min(HISTORY_MAX),
        None => 0,
    }
}

/// Yields legal moves best-first, one selection pass per `next()`.
/// Order is identical to a stable descending sort by `move_order_score`.

struct OrderedMoves {                     // no lifetime param any more
    moves: MoveList,
    scores: ArrayVec<i32, 270>,
    next: usize,
}

impl OrderedMoves {
    fn new(pos: &Chess, hash_move: PackedMove, killers: [Option<Move>; 2],
           history: &[[AtomicI32; 64]; 64]) -> Self {
        Self::from_list(pos.legal_moves(), hash_move, killers, history)
    }

    fn from_list(moves: MoveList, hash_move: PackedMove, killers: [Option<Move>; 2],
                 history: &[[AtomicI32; 64]; 64]) -> Self {
        let scores = moves.iter()
            .map(|m| move_order_score(*m, hash_move, killers, history))
            .collect();
        Self { moves, scores, next: 0 }
    }

    fn len(&self) -> usize { self.moves.len() }
}

impl Iterator for OrderedMoves {
    type Item = Move;

    fn next(&mut self) -> Option<Move> {
        let i = self.next;
        if i >= self.moves.len() { return None; }

        let mut best = i;
        for j in (i + 1)..self.moves.len() {
            if self.scores[j] > self.scores[best] { best = j; }
        }
        self.moves[i..=best].rotate_right(1);
        self.scores[i..=best].rotate_right(1);   // keep them in lockstep
        self.next += 1;
        Some(self.moves[i])
    }
}


/// Whether the position hashing to `key` has already occurred on the path that
/// leads to it.
///
/// `keys` holds the positions before the current one, oldest first, so the slot
/// the current position *would* take is `keys.len()`. Only every second entry can
/// match, because a position can only recur with the same side to move.
///
/// `halfmoves` bounds how far back it is worth looking. Captures and pawn moves
/// are the only irreversible moves in chess and both reset that clock, so no
/// position older than the last one of them can ever come back. `floor` is the
/// null-move barrier; the scan stops at whichever of the two bounds is tighter.
///
/// A single earlier occurrence is treated as a draw rather than waiting for a
/// third. The side steering towards the repetition can nearly always force the
/// third occurrence once it has the second, so insisting on it would only find
/// the same draw two plies later and two plies deeper.
fn is_repetition(keys: &[Zobrist64], key: Zobrist64, halfmoves: u32, floor: usize) -> bool {
    let mut i = keys.len();
    let limit = i.saturating_sub(halfmoves as usize).max(floor);
    while i >= limit + 2 {
        i -= 2;
        if keys[i] == key {
            return true;
        }
    }
    false
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

fn record_killer(ctx: &mut Worker, ply: usize, m: Move) {
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

