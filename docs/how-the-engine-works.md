# How This Chess Engine Works

A guided tour of every technique in the engine, written to be read top to bottom.
Early sections build the foundations; later sections each explain one specific
technique, why it exists, and where it lives in the code. Nothing here requires
prior chess-programming knowledge — only the basics of chess itself.

---

## 1. The big picture

The project builds **two binaries** on top of one engine core:

* `chess-engine` (`src/main.rs`) — play against the engine in a terminal UI.
* `uci` (`src/bin/uci.rs`) — speak the **UCI** protocol on stdin/stdout, so the
  engine can be plugged into any chess GUI (Cute Chess, Arena, …).

Both drive the same brain through the same trait:

```
┌─────────────┐   Frontend trait    ┌──────────────┐
│  src/tui.rs │ ◄─────────────────► │ src/play.rs  │ ──┐
│ (terminal   │  choose_side,       │ (human-vs-   │   │
│  UI)        │  request_move, show,│  engine loop)│   │  ChessEngine trait
└─────────────┘  thinking, game_over└──────────────┘   │  set_position, play_move,
                                                       ├─►  search(on_progress),
┌──────────────────────┐                               │   set_hash_size_mb,
│ src/uci/client.rs    │ ──────────────────────────────┘   clear, search_handle
│ (UCI protocol loop,  │                               ┌──────────────┐
│  threads + channels) │                               │ src/engine/  │
└──────────────────────┘                               │ (search+eval)│
                                                       └──────────────┘
```

* `src/engine/` — the chess brain:
  * `search.rs` — `SearchEngine`, negamax, quiescence, iterative deepening.
  * `eval.rs` — static evaluation (material + piece-square tables).
  * `tt.rs` — the transposition table.
  * `interface.rs` — the `ChessEngine` trait and the plain data types that cross
    the boundary (`Limits`, `TimeMode`, `SearchProgress`, `SearchResult`,
    `SearchHandle`, `Score`).
* `src/play.rs` — a small renderer-agnostic game loop for human-vs-engine. It
  alternates turns, records SAN history, and builds the PGN at the end. It talks
  only to the `Frontend` and `ChessEngine` traits, so both sides are swappable.
* `src/tui.rs` — one concrete `Frontend`: a crossterm terminal UI with arrow-key
  move selection and highlighted squares.
* `src/uci/client.rs` — the UCI front end. It does *not* use `play.rs`; a GUI
  drives the game, so this layer only translates protocol messages into
  `ChessEngine` calls (§12).
* `src/render.rs` / `src/render/terminal.rs` — a tiny plain-text board printer,
  handy for debugging; the TUI draws its own board and does not use it.

Board representation, move generation, and rules all come from the
[`shakmaty`](https://docs.rs/shakmaty) crate: `Chess` is a position,
`legal_moves()` generates moves, `play()`/`play_unchecked()` apply them,
`is_game_over()` / `outcome()` detect the end, and `zobrist_hash()` gives the
position fingerprint the transposition table is keyed on. We never implement
chess rules ourselves.

The engine's job reduces to one question, asked once per turn:

> **Given this position, which legal move leads to the best future?**

Everything below is about answering that question quickly and accurately.

---

## 2. The engine boundary (`engine/interface.rs`)

Before the algorithms, the shape of the API they sit behind — because a few
design choices in the search only make sense in its light.

```rust
pub trait ChessEngine {
    fn set_position(&mut self, pos: Chess);
    fn play_move(&mut self, mv: Move) -> Result<(), IllegalMove>;
    fn search(&mut self, on_progress: &mut dyn FnMut(&SearchProgress)) -> SearchResult;
    fn set_hash_size_mb(&mut self, mb: usize);
    fn clear(&mut self);
    fn search_handle(&self) -> SearchHandle;
}
```

Two things are worth noticing:

**The engine owns the position.** `search()` takes no position argument; it
searches whatever was last given via `set_position` / `play_move`. Callers keep
their own copy for display purposes, and both front ends push every played move
into the engine as well.

**Search limits arrive out-of-band, through a `SearchHandle`.** The handle is an
`Arc` over a stop flag (`AtomicBool`), a `Mutex<Limits>`, and a sequence counter
(`AtomicU64`), so it can be cloned and used from *another thread* while a search
is running. `Limits` says what "stop" means:

```rust
pub struct Limits {
    pub time_mode: TimeMode,          // Unbound | Fixed(Duration) | Clock(Clock)
    pub max_depth: Option<u8>,
    pub max_nodes: Option<u64>,
    pub restrict_to: Option<Vec<Move>>,   // UCI `searchmoves`
}
```

Each `set_limits` bumps `limits_seq`. The search notices the bump mid-flight
(§7) and recomputes its deadline — which is exactly how UCI `ponderhit` turns an
unbounded ponder search into a timed one without restarting it.

`SearchProgress` (one per completed depth, delivered to the `on_progress`
callback) and `SearchResult` (the final answer) are plain snapshots: depth,
score, nodes, PV, elapsed time, hash fullness.

---

## 3. Evaluation — putting a number on a position (`eval.rs`)

A search must eventually stop and *judge* the positions it reaches. That judge
is `evaluate(pos)`, which returns a score in **centipawns** (100 = one pawn of
advantage), always **relative to the side to move** (positive = good for
whoever's turn it is — this sign convention is what negamax expects, see §4).

The evaluation has two ingredients:

### Material

Each piece has a classical value: pawn 100, knight 320, bishop 330, rook 500,
queen 900 (the king is 0 — it is never captured). This is the dominant term — no
positional bonus is allowed to outweigh real material, which keeps the engine
from sacrificing pieces for "pretty" squares.

### Piece-square tables (PSTs)

A 64-entry table per piece type saying "this piece earns this bonus/penalty on
this square" (the well-known Michniewski *Simplified Evaluation* tables). They
encode positional common sense that pure material can't:

* Knights: strong in the center (+20), bad on the rim (−50).
* Pawns: rewarded for advancing and holding the center.
* Bishops: like long diagonals, hate corners.
* Rooks: bonus on the 7th rank.
* King: **two** tables, see below.

The tables are written **rank 8 first**, so table index 0 is a8 and index 63 is
h1. Consequently a *White* piece must have its square's rank flipped
(`sq ^ 56`), while a *Black* piece indexes the table with its square directly —
which mirrors it vertically. That is the symmetry we want: a black piece on e7
scores like a white piece on e2. Both cases live in `pst_index`, so one table
serves both colors.

An important mental model: a PST is a **prior**, not a rule. It only decides
between positions the search can't tell apart concretely. If the search can
*see* a knight-on-the-rim fork winning a rook within its depth, the +500
material at that leaf swamps the −40 square penalty. The PST never vetoes a
line and never affects move ordering — it only nudges leaf scores.

### Tapered king evaluation

The king wants opposite things in different game phases: hide behind pawns in
the middlegame, march to the center in the endgame. So there are two king
tables (`KING_MG`, `KING_EG`) blended by a **phase counter**: queens count 4,
rooks 2, minors 1, summed over the board (24 at the start, → 0 as pieces come
off):

```
king_score = (KING_MG[sq] * phase + KING_EG[sq] * (24 - phase)) / 24
```

The king's evaluation *slides smoothly* from "castle and hide" to "centralize"
as material leaves the board, with no abrupt switch.

---

## 4. Negamax — the search skeleton (`search.rs::negamax`)

Chess is a tree: from the current position each legal move leads to a child
position, and so on. **Minimax** picks the move that maximizes your result
assuming the opponent always replies with *their* best move.

**Negamax** is minimax with a notational trick: since chess is zero-sum, "good
for me" = "bad for you," so instead of alternating max/min layers, every node
maximizes and we **negate the child's score**:

```
score = -negamax(child_position, depth - 1, ...)
```

This works precisely because `evaluate()` is side-to-move-relative. One
function, no White/Black special cases anywhere in the search.

* `depth` — how many plies (half-moves) we still intend to search below here.
  At `depth == 0` we stop and evaluate (via quiescence, §6).
* `ply` — how far below the *root* we are. Used for mate scores, killer slots,
  the root-only `searchmoves` filter, and the check-extension cap.

The function returns `Option<i32>`: `None` means **"this search was aborted"**
(time/node limit or a `stop` command). `None` propagates all the way up through
every `?`, so a timed-out iteration discards its partial results entirely.

The function also builds the **PV (principal variation)** — the actual best line
of play — through a `&mut Vec<Move>` parameter: whenever a move improves alpha,
the node sets its PV to `[that move] + child's PV`. The move the engine plays is
simply `pv[0]`. (Reconstructing the PV from the transposition table instead
doesn't work reliably — entries get overwritten — which is why it's threaded
through the recursion explicitly.)

### Mate scores

Checkmate is scored as `MATE - ply` (`MATE = 1,000,000`), so a mate found
*sooner* scores *higher* — the engine prefers the fastest mate and, when
losing, the slowest one. Any score above `MATE_THRESHOLD` (4000 = the total
material on a full board, unreachable by normal evaluation) is recognized as
"this is a forced mate," which several techniques use as a guard. `mate_in()`
converts such a score back into a distance in *moves* for display and for UCI's
`score mate n`.

Terminal positions are scored by `leaf_score`: checkmate → ±(MATE − ply),
stalemate or insufficient material → 0.

---

## 5. Alpha-beta pruning — skipping refuted branches

Plain negamax visits the whole tree: ~35^depth nodes. Alpha-beta gets the
**same answer** while skipping most of it.

The search carries a window `(alpha, beta)`:

* `alpha` — the best score the side to move is already guaranteed elsewhere.
* `beta` — the score at which the *opponent* would refuse to enter this line
  (they already have something better).

When a move scores `>= beta`, we have a **beta cutoff**: this node is already
"too good," the opponent will never allow it, so the remaining sibling moves
are irrelevant and we stop the loop. This is *sound* — pruned moves provably
cannot change the result.

The catch: the savings depend almost entirely on **move order**. Find the best
move first → immediate cutoff → siblings skipped. Find it last → you searched
everything anyway. With perfect ordering alpha-beta searches ~√(the full tree).
That's why a large share of this engine (§9) is dedicated purely to guessing
good moves early.

---

## 6. Quiescence search — don't evaluate mid-capture (`quiescence`)

If the search stops at `depth == 0` right after `QxP`, the evaluation sees
"I'm up a pawn!" — not noticing the queen is hanging and gets recaptured next
move. This is the **horizon effect**: judging a position mid-tactic.

Quiescence search fixes it: at depth 0, instead of evaluating immediately, keep
searching **captures and promotions only** (`Move::is_conversion()`) until the
position is quiet, then evaluate. Two details:

* **Stand-pat**: the static eval acts as a floor — you're allowed to *not*
  capture. If standing pat already beats beta, cut off immediately. This keeps
  quiescence from being forced into bad captures.
* Since only conversions are explored, the tree shrinks rapidly (captures run
  out), so it terminates on its own.

Three things quiescence deliberately does *not* do, all of which are normal in
stronger engines: it has no depth cap, it doesn't probe or store in the
transposition table, and it doesn't prune losing captures (no SEE, §15). It also
never checks the clock — the time check lives in `negamax` only — so a long
capture sequence can overrun the deadline by a small margin.

Note that an in-check node never reaches quiescence at all: the check extension
(§10d) raises its depth before the `depth == 0` test, so we never statically
evaluate a position whose king is attacked.

---

## 7. Iterative deepening and time management (`SearchEngine::search`)

`search()` is the entry point: it resets the per-search context (`reset_ctx`),
then searches the same position at depth 1, 2, 3, … up to `MAX_DEPTH` (255),
each iteration a full `negamax` from the root (`depth_bound_search`).

Sounds wasteful — it isn't, because trees grow exponentially: all the shallow
iterations together cost a fraction of the deepest one. In exchange we get:

* **Time management**: the engine always has a complete answer from the last
  finished depth. Only a *completed* iteration commits its PV, so a timed-out
  iteration is thrown away.
* **Warm-up for move ordering**: each iteration fills the transposition table
  with best moves that the next, deeper iteration tries first (§9). This is the
  main reason iterative deepening is a net *win* rather than a cost.
* **Progress reporting**: after each completed depth, `on_progress` fires with
  depth / score / nodes / PV / elapsed / hashfull — the TUI shows it live, the
  UCI layer turns it into an `info` line.

The loop also stops early when the score is a forced mate (`|score| >=
MATE_THRESHOLD`), and when `limits.max_depth` is reached.

### Stopping (`should_stop`)

Inside `negamax`, every `CHECK_STOP_AFTER` (2000) nodes the search asks whether
it should stop. It stops when any of these is true:

* the deadline has passed,
* `limits.max_nodes` has been reached,
* the `SearchHandle`'s stop flag was set (UCI `stop` / `quit`).

Two details:

* The check is skipped while `search_ctx.depth == 0`, i.e. during the very first
  (depth 1) iteration. That guarantees at least one completed iteration and
  therefore always a legal move to return, even with an absurdly short budget.
* `should_stop` compares the handle's `limits_seq` with its own copy and
  re-reads `Limits` when it changed, recomputing the deadline from *now*. This
  is what makes `ponderhit` work mid-search.

### Deadlines (`calc_deadline`)

```
TimeMode::Unbound     → no deadline (runs until `stop`, or a mate is found)
TimeMode::Fixed(d)    → now + d
TimeMode::Clock(c)    → base = c.remaining / c.moves_to_go        (if given)
                             = c.remaining / 25                   (otherwise)
                             + c.increment * 3/4
                        cap  = (c.remaining - 100ms) / 2
                        now + min(base, cap)
```

The `/ 25` assumes roughly 25 moves left to play; the cap is a safety valve so a
single move can never eat more than half the remaining clock.

Because `Unbound` has no deadline, `search()` ends with a small sleep-loop after
the deepening loop: `go infinite` and `go ponder` must *not* answer until `stop`
or `ponderhit` arrives, even if the deepening loop ran out of depth or found a
mate.

---

## 8. The transposition table (`tt.rs`)

The same position is reached via many different move orders. The TT caches, for
each searched position:

* the `depth` it was searched to,
* the `score`, and what kind of score it is (`bound`),
* the `best_move` found.

It is a fixed-size, **direct-mapped** table (one slot per index, no buckets)
keyed by the position's **Zobrist hash**, a 64-bit fingerprint from shakmaty.
The slot index is the low bits of the key (`key & mask`), so the entry count is
always a power of two — `entry_count` rounds the requested MiB *down* to a power
of two (never below `MIN_ENTRIES = 1024`, so a tiny `Hash` setting still works).

On revisiting a position, if the stored entry is deep enough, we can sometimes
return its score instantly instead of re-searching an entire subtree.

**Bounds** exist because alpha-beta rarely computes exact scores. If a node
never raised alpha, we only learned "score ≤ X" (**Upper** bound); if it got a
beta cutoff, only "score ≥ X" (**Lower** bound); only in between is it
**Exact**. A stored bound is usable only when it still decides the current
`(alpha, beta)` question — the probe check in `negamax`:

```rust
Bound::Exact => true,
Bound::Lower => e.score >= beta,     // proves a cutoff
Bound::Upper => e.score <= alpha,    // proves a fail-low
```

Four subtleties:

* **Replacement**: `store` keeps the existing entry only when it is the *same
  position* searched *deeper*; anything else (a different position hashing to
  the same slot, or a deeper re-search of the same one) overwrites. Simple
  "always replace except deeper-same-key", which suits iterative deepening
  because later iterations are the deeper ones.
* **Key 0 is never cached.** `key == 0` doubles as the "empty slot" sentinel,
  which lets every slot be a plain `TtEntry` instead of an `Option<TtEntry>`
  (more entries per MiB). Both `probe` and `store` skip it — one position in
  2^64 simply goes uncached.
* **Mate scores** are stored ply-adjusted (`score_to_tt` / `score_from_tt`), so
  "mate in 3 from here" stays correct when the same position is reached at a
  different distance from the root.
* **Never take a TT cutoff at the root** (`ply == 0`): a cutoff returns before
  the move loop runs, i.e. with an empty PV — and at the root, no PV means no
  move to play. Currently unreachable in practice (the root is searched with a
  full window at ever-increasing depth) but cheap insurance for the day
  aspiration windows are added.

`filled` is maintained by `store`, so `fill_fraction()` is O(1) — that is what
becomes UCI's `hashfull` (in permille). `clear()` wipes the table (UCI
`ucinewgame`); `resize()` rebuilds it for a new `Hash` option, dropping the old
allocation before making the new one so two large tables never coexist.

---

## 9. Move ordering — guessing the best move first (`ordered_moves`)

Everything from §5 onward lives or dies on move order, so all legal moves are
scored and sorted into explicit tiers (highest first):

| Tier | Score | What |
|---|---|---|
| Hash move | 1,000,000 | The TT's stored best move for this position |
| Captures / promotions | 100,000 + MVV-LVA (capture) and/or 10 × promoted piece | The tactical tier |
| Killer 1, 2 | 90,000 / 80,000 | Quiet moves that recently caused cutoffs at this ply |
| Quiet moves | history value, capped at 70,000 | Ordered by past success anywhere in the tree |

The gaps between the tiers are wide enough that they can never overlap — hence
the 70,000 cap on history.

* **Hash move**: even when a TT entry's *score* can't be used, its *move* is
  the single best guess available — it was literally proven best here before,
  usually by the previous iterative-deepening iteration. This is the main
  channel through which each iteration teaches the next.
* **MVV-LVA** (`10 × victim − attacker`): try QxR before PxP before RxP. Cheap
  and effective for ordering captures. A promotion adds `10 × promoted piece` on
  top, so queening (and capturing while queening) sorts to the front.
* **Killer moves** (2 slots per ply): sibling positions at the same ply are
  similar, so a quiet move that refuted one sibling (caused a beta cutoff)
  often refutes the others — e.g. the same knight-fork square. Stored per ply,
  most recent first, and **cleared at the start of every search**.
* **History heuristic** (`history[from][to] += depth²` on quiet cutoffs): a
  global "this move keeps being good" table that orders the remaining quiet
  moves. Deeper cutoffs weigh more because they were more expensive to find.
  Unlike killers, the history table is *not* cleared between searches — it ages
  only by being outgrown (see §15).

Killers and history matter beyond their direct (modest) node savings: they are
what makes "late move = probably bad move" true enough for LMR (§10b) to be safe.

Quiescence calls the same `ordered_moves` (with no hash move and no killers) and
then filters to conversions — simple, at the cost of generating and sorting
quiet moves it will never search.

### `searchmoves`

At the root only (`ply == 0`), if `limits.restrict_to` is set, the move list is
filtered down to those moves. This implements UCI's `go searchmoves …`.

---

## 10. Searching *selectively* — spending depth where it matters

A fixed-depth search wastes most of its effort on hopeless moves. The next
four techniques bend the tree: promising/forcing lines get searched deeper,
unpromising ones shallower.

### 10a. Principal Variation Search (PVS)

After the first move at a node establishes alpha, we don't need *exact* scores
for the siblings — only proof they're **worse**. That yes/no question is asked
with a **null window** `(alpha, alpha+1)`: a window with no room inside it, so
the child search fails low or fails high almost immediately, with maximal
pruning either way.

```
first move : full window  (-beta, -alpha)          — full effort
later moves: scout        (-alpha-1, -alpha)       — cheap "is it worse?"
             if the scout beats alpha inside the window → re-search fully
```

The occasional re-search is the price; with good ordering it's rare, and the
scouts are far cheaper than full searches. Only the PV itself gets full-window
treatment — hence the name.

### 10b. Late Move Reductions (LMR)

Given good ordering, a *quiet* move that is 4th, 5th, 10th in the list is very
unlikely to be best. LMR searches such moves at **reduced depth** first
(`depth − 1 − r`).

```rust
r = 0.75 + ln(depth) * ln(move_count) / 2.25     // clamped to [1, depth - 2]
```

so `r` grows slowly with both remaining depth and how late the move is, and the
reduced search always keeps at least one ply.

Guards — a move is reduced only when all of these hold: `depth >= 3`, it is at
least the 4th move (`move_count >= 3`), it is quiet (no capture, no promotion),
the node is not in check, and the move does not give check. Forcing and
tactical moves always get full depth.

The safety net: if a reduced move *beats alpha anyway*, it is immediately
**re-searched at full depth** (still with a null window) and, if it also lands
inside `(alpha, beta)`, once more with the full window — the PVS machinery. So a
wrongly-reduced move costs one or two extra searches; it can never sneak into
the PV on the strength of a shallow score alone.

### 10c. Null-move pruning (verified)

The observation: in chess, having the move is almost always an advantage. So
ask: *"if I pass (do nothing) and my opponent moves twice, am I still doing
too well (≥ beta)?"* If yes, the real position — where I do get a move — is
surely a cutoff, so prune the whole node without searching my moves.
Implemented with shakmaty's `swap_turn()` and a reduced-depth null-window search
around beta, with `R = 2` (`3` from depth 6 up).

Preconditions: `depth >= 3`, not in check (passing is meaningless there), `beta`
is not a mate score (a pass can never prove a mate), and the side to move has at
least one non-pawn piece.

**The flaw — zugzwang**: positions where every real move *loses* but passing
would be fine (trapped piece, king-and-pawn endgames). There "pass ≥ beta" says
nothing about real moves, and naive null-move prunes a lost position as won.
Zugzwang is a property of the whole move tree, so there is no cheap static
detector for it. Three defenses in this engine:

1. Disabled when the side to move has only king + pawns (`has_non_pawn_material`),
   where zugzwang is routine.
2. **No two null moves in a row** (`allow_null` flag) — a double pass is a
   degenerate non-position.
3. **Verification**: when the null search fails high, we do *not* prune
   immediately. We re-search the node at reduced depth (`depth − R`) with **real
   moves only** (null disabled). Genuine zugzwang → the verification fails low →
   we refuse to prune and search normally. The verification search is exactly the
   zugzwang detector that can't be written statically: it plays the moves.

Only if the verification also reaches beta does the node return `beta`
immediately.

### 10d. Check extensions

The mirror image of reductions: when the side to move is **in check**, search
one ply *deeper* (`depth += 1`), before the `depth == 0` test. Checks are
forcing — few legal replies, and tactics (mating attacks, perpetuals) hide
behind them — so the extra depth goes exactly where the tree is narrow and
critical. It also guarantees we never statically evaluate an in-check position
(whose eval would be garbage — quiescence only explores captures, not check
evasions).

**The recursion trap**: +1 per check means depth never decreases along a
line of consecutive checks. Since the engine has no repetition detection yet,
a perpetual check would recurse forever and overflow the stack. The guard
`ply < MAX_PLY (128)` stops extending past that depth, bounding the recursion.
(Proper repetition detection is the "real" fix and a good future addition.)

---

## 11. How a move actually gets chosen — the full pipeline

Putting it all together, one engine move works like this:

```
search()                                           ← position already set
├─ reset_ctx: read limits, compute deadline, clear killers/PV/node count
└─ iterative deepening: depth = 1, 2, 3, … (stop on limit, abort, or mate)
   └─ negamax(pos, depth, -inf, +inf, ply=0)
      ├─ count node, clear this node's PV
      ├─ game over? → leaf score (mate ± ply, or 0)
      ├─ in check (and ply < 128)? → depth += 1            (check extension)
      ├─ depth == 0 → quiescence (conversions only, stand-pat)
      ├─ every 2000 nodes: out of time/nodes/stopped? → None (abort)
      ├─ TT probe → maybe instant cutoff (never at the root); remember hash move
      ├─ null-move: pass, reduced search; fail-high → verify → maybe prune
      ├─ order moves: hash > MVV-LVA/promotions > killers > history
      │  (at the root: filter by `searchmoves` if given)
      └─ for each move:
         ├─ 1st move: full window, full depth
         ├─ later moves: reduced depth if quiet + late (LMR), null window (PVS)
         │   └─ beats alpha? → re-search at full depth, then full window
         ├─ raises alpha? → PV = [move] + child PV
         ├─ beta cutoff? → if quiet: record killer + history += depth²; break
         └─ store {depth, score, bound, best move} in the TT
   └─ after each completed depth: commit the PV, fire on_progress
      (TUI shows "depth N, score, kn/s"; UCI prints an `info` line)
└─ (Unbound only) wait for `stop`/`ponderhit`
└─ SearchResult { best_move: pv[0], pv, score, depth, nodes }
```

---

## 12. The UCI front end (`uci/client.rs`)

`run_with(input, output)` is the whole protocol loop, built from **three
threads** joined by channels:

1. a **reader** thread turning stdin lines into `Event::Line`,
2. an **engine** thread owning the `SearchEngine` and consuming `EMessage`
   commands (`Search`, `SetPosition`, `PlayMove`, `SetHashSize`, `Clear`),
3. the **main** loop, which parses messages (via `vampirc-uci`), sends work to
   the engine thread, and writes every `info` / `bestmove` line.

The split is what makes `stop` work: the engine thread is busy searching, the
main loop stays responsive, and `stop` just flips the atomic in the shared
`SearchHandle` (§2) — no interruption machinery needed.

Supported commands: `uci`, `isready`, `ucinewgame` (clears the TT),
`position [startpos|fen] moves …`, `go` (`infinite`, `ponder`, `movetime`,
`wtime/btime/winc/binc/movestogo`, `depth`, `nodes`, `searchmoves`), `stop`,
`ponderhit`, `setoption name Hash value N` (clamped to 1–4096 MiB), and `quit`.
Options advertised: `Hash` (spin, default 16) and `Ponder` (check).

**Pondering** works like this: `go ponder` sets `TimeMode::Unbound` on the
handle but *remembers* the real limits from the same `go` line. When
`ponderhit` arrives, those stored limits are installed, the sequence counter
bumps, and the running search picks up a real deadline (§7) instead of
restarting. When a search finishes, `bestmove <m> ponder <pv[1]>` is printed —
or `bestmove 0000` when there is no move.

Known rough edges in the reporting: `nodes_per_s` is always sent as 0, and
`seldepth` is reported as the nominal depth (real selective depth, including
extensions and quiescence, is not tracked yet).

---

## 13. The terminal front end and PGN (`tui.rs`, `play.rs`)

`play_game` asks the front end which color the human takes, then alternates:
`request_move` for the human, `search` for the engine, pushing every move into
both its own `Chess` position and the engine's.

* **Live thinking display**: because iterative deepening naturally pauses
  between depths, a plain synchronous callback after each finished depth gives
  smooth updates with no threads and no races. The TUI prints depth, score,
  nodes and kn/s — but **deliberately not the PV**, which would reveal the
  engine's expected line, including the human's best replies.
* **Move picking**: arrow keys move a cursor, `Enter` selects a piece and lights
  up its legal targets, `Enter` again plays, `Backspace` cancels, `q`/`Esc`
  quits. Promotions get their own little picker. The front end only ever returns
  legal moves, which is why the game loop can `expect()` them.
* **PGN export** (`play.rs::to_pgn`): the loop records every move in SAN
  (`SanPlus`, so `+`/`#` suffixes are right) and, at game end, emits a standard
  seven-tag PGN wrapped at 80 columns, which the TUI offers to save.

`src/main.rs` wires this up with a **1024 MiB** transposition table and a fixed
**2 seconds** per move, and installs a panic hook that restores the terminal
before the default handler prints.

---

## 14. Tests

* `tests/epd_tests.rs` — tactical positions in EPD format (`bm` = best move)
  from `tests/positions/`. Each is searched with a 16 MiB table, a 1 s deadline
  **and** a depth cap of 4, and the engine's move must match the `bm`. This is
  the regression suite search changes are validated against. The `run_scenario`
  test is an `#[ignore]`d single-file runner:
  `SCENARIO=<file>.epd cargo test run_scenario -- --ignored --nocapture`.
* `tests/nodes_bench.rs` (`#[ignore]`d) — "how deep in 2 seconds?" on the
  Kiwipete position, with a 500 MiB table. Used to A/B each technique:
  `cargo test --release --test nodes_bench -- --ignored --nocapture`.
* Unit tests live next to the code: `eval.rs` (symmetry, PST sanity),
  `tt.rs` (store/probe, the key-0 sentinel, depth-preferred replacement, sizing),
  `play.rs` (PGN roster, movetext, date math), `tui.rs` (cursor math, targets).

### Measured impact

These are the A/B numbers recorded when each technique was added (same hardware,
~2 s per move, Kiwipete). They are historical: they show why each piece is in
the engine, not a current benchmark.

| Change | Effect |
|---|---|
| Hash-move ordering | ~30% fewer nodes under iterative deepening |
| Killers + history | small alone (~2%), enables LMR |
| PVS | ~3–5% fewer nodes, grows with depth |
| **LMR** | **depth 8 → 12 in the same time** — the biggest single win |
| Verified null-move | +3 plies in quiet positions, zugzwang-safe |
| Check extensions | deeper forcing lines (not visible in nominal depth) |
| PST + tapered eval | plays real openings (Scotch in self-play) instead of shuffling |

---

## 15. What's deliberately *not* here (future work)

Roughly in order of value:

1. **Repetition & 50-move detection** — the engine can't recognize threefold
   repetition or the 50-move rule (shakmaty's `is_game_over` covers only
   checkmate, stalemate and insufficient material), so it neither claims such
   draws nor avoids them when winning. Also the proper fix for the
   perpetual-check recursion cap (§10d).
2. **Aspiration windows** — start each iteration with a narrow window around
   the previous score instead of (−∞, +∞); re-search wider on failure. (This
   is also what makes the root-TT-cutoff guard from §8 load-bearing.)
3. **Static Exchange Evaluation (SEE)** — evaluate capture sequences on one
   square without searching; used to skip losing captures in quiescence and
   order captures better than MVV-LVA.
4. **Richer evaluation** — mobility, passed pawns, king-safety terms, bishop
   pair.
5. **Search-report polish** — real `seldepth` tracking and an actual
   nodes-per-second figure in `SearchProgress` (both currently placeholders),
   plus history aging between searches.
6. **Quiescence tuning** — generate only conversions instead of filtering the
   full legal move list, and add a depth cap.
7. **Opening book / endgame tablebases.**
8. **Multithreading (Lazy SMP)** — worth ~2–3× nodes. The UCI layer already runs
   the search on its own thread, but the search itself is single-threaded;
   intentionally postponed because all of the above were worth more.
