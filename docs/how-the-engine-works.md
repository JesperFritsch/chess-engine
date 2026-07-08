# How This Chess Engine Works

A guided tour of every technique in the engine, written to be read top to bottom.
Early sections build the foundations; later sections each explain one specific
technique, why it exists, and where it lives in the code. Nothing here requires
prior chess-programming knowledge — only the basics of chess itself.

---

## 1. The big picture

The program is split into three layers that deliberately know very little about
each other:

```
┌────────────┐     Frontend trait      ┌────────────┐    ChessEngine trait    ┌────────────┐
│  src/tui.rs │ ◄─────────────────────► │ src/play.rs │ ◄─────────────────────► │ src/engine/ │
│  (terminal  │   choose_side,          │  (the game  │   best_move(pos,        │  (search +  │
│   UI)       │   request_move, show,   │   loop)     │   progress callback)    │   eval)     │
└────────────┘   thinking, game_over    └────────────┘                         └────────────┘
```

* `src/engine/` — the actual chess brain: `search.rs` (finding the best move),
  `eval.rs` (judging a position), `tt.rs` (the transposition table).
* `src/play.rs` — a small renderer-agnostic game loop. It alternates turns,
  validates nothing itself (moves arrive already-legal), records SAN history,
  and builds the PGN at the end. It talks only to two traits, so both the UI
  and the engine are swappable.
* `src/tui.rs` — one concrete `Frontend`: a crossterm terminal UI with
  arrow-key move selection and highlighted squares.

Board representation, move generation, and rules all come from the
[`shakmaty`](https://docs.rs/shakmaty) crate: `Chess` is a position,
`legal_moves()` generates moves, `play()` applies them, `is_game_over()` /
`outcome()` detect the end. We never implement chess rules ourselves.

The engine's job reduces to one question, asked once per turn:

> **Given this position, which legal move leads to the best future?**

Everything below is about answering that question quickly and accurately.

---

## 2. Evaluation — putting a number on a position (`eval.rs`)

A search must eventually stop and *judge* the positions it reaches. That judge
is `evaluate(pos)`, which returns a score in **centipawns** (100 = one pawn of
advantage), always **relative to the side to move** (positive = good for
whoever's turn it is — this sign convention is what negamax expects, see §3).

The evaluation has two ingredients:

### Material

Each piece has a classical value: pawn 100, knight 320, bishop 330, rook 500,
queen 900. This is the dominant term — no positional bonus is allowed to
outweigh real material, which keeps the engine from sacrificing pieces for
"pretty" squares.

### Piece-square tables (PSTs)

A 64-entry table per piece type saying "this piece earns this bonus/penalty on
this square." They encode positional common sense that pure material can't:

* Knights: strong in the center (+20), bad on the rim (−50).
* Pawns: rewarded for advancing and holding the center.
* Bishops: like long diagonals, hate corners.
* Rooks: bonus on the 7th rank.
* King: **two** tables, see below.

The tables are written from White's perspective; a Black piece indexes the
table vertically mirrored (`sq ^ 56` trick in `pst_index`), so one table serves
both colors symmetrically.

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
king_score = (MG[sq] * phase + EG[sq] * (24 - phase)) / 24
```

The king's evaluation *slides smoothly* from "castle and hide" to "centralize"
as material leaves the board, with no abrupt switch.

---

## 3. Negamax — the search skeleton (`search.rs::negamax`)

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
  At `depth == 0` we stop and evaluate (via quiescence, §5).
* `ply` — how far below the *root* we are. Used for mate scores, killers, and
  the check-extension cap.

The function also builds the **PV (principal variation)** — the actual
best line of play — through a `&mut Vec<Move>` parameter: whenever a move
improves alpha, the node sets its PV to `[that move] + child's PV`. The move
the engine plays is simply `pv[0]`. (Reconstructing the PV from the
transposition table instead doesn't work reliably — entries get overwritten —
which is why it's threaded through the recursion explicitly.)

### Mate scores

Checkmate is scored as `MATE - ply` (`MATE = 1,000,000`), so a mate found
*sooner* scores *higher* — the engine prefers the fastest mate and, when
losing, the slowest one. Any score above `MATE_THRESHOLD` (≈ total material
value, unreachable by normal evaluation) is recognized as "this is a forced
mate," which several techniques use as a guard.

---

## 4. Alpha-beta pruning — skipping refuted branches

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
That's why a large share of this engine (§7) is dedicated purely to guessing
good moves early.

---

## 5. Quiescence search — don't evaluate mid-capture (`quiescence`)

If the search stops at `depth == 0` right after `QxP`, the evaluation sees
"I'm up a pawn!" — not noticing the queen is hanging and gets recaptured next
move. This is the **horizon effect**: judging a position mid-tactic.

Quiescence search fixes it: at depth 0, instead of evaluating immediately, keep
searching **captures/promotions only** until the position is quiet, then
evaluate. Two details:

* **Stand-pat**: the static eval acts as a floor — you're allowed to *not*
  capture. If standing pat already beats beta, cut off immediately. This keeps
  quiescence from being forced into bad captures.
* Since only captures are explored, the tree shrinks rapidly (captures run
  out), so it terminates fast.

---

## 6. Iterative deepening + the transposition table

These two work as a pair; each makes the other better.

### Iterative deepening (`time_bound_search`)

Instead of guessing "search to depth N," search depth 1, then 2, then 3, …
until the time budget runs out. Sounds wasteful — it isn't, because trees grow
exponentially: all the shallow iterations together cost a fraction of the
deepest one. In exchange we get:

* **Time management**: the engine always has a complete answer from the last
  finished depth. A deadline check runs every ~2000 nodes; when time is up the
  current iteration aborts (returns `None` up the whole stack) and its partial
  results are discarded — only fully completed iterations count.
* **Warm-up for move ordering**: each iteration fills the transposition table
  with best moves that the next, deeper iteration tries first (§7).
* **Progress reporting**: after each completed depth, a callback fires with
  depth / score / node count, which the TUI displays live (§10).

### Transposition table (`tt.rs`)

The same position is reached via many different move orders. The TT is a big
fixed-size hash table (keyed by the position's **Zobrist hash**, a 64-bit
fingerprint provided by shakmaty) caching, for each searched position:

* the `depth` it was searched to,
* the `score`, and what kind of score it is (`bound`),
* the `best_move` found.

On revisiting a position, if the stored entry is deep enough, we can sometimes
return its score instantly instead of re-searching an entire subtree.

**Bounds** exist because alpha-beta rarely computes exact scores. If a node
never raised alpha, we only learned "score ≤ X" (**Upper** bound); if it got a
beta cutoff, only "score ≥ X" (**Lower** bound); only in between is it
**Exact**. A stored bound is usable only when it still decides the current
`(alpha, beta)` question — the `probe` logic checks this.

Two subtleties:

* **Mate scores** are stored ply-adjusted (`score_to_tt`/`score_from_tt`), so
  "mate in 3 from here" stays correct when the same position is reached at a
  different distance from the root.
* **Never take a TT cutoff at the root** (`ply == 0`): a cutoff returns before
  the move loop runs, i.e. with an empty PV — and at the root, no PV means no
  move to play. Currently unreachable in practice (the root is searched with a
  full window at ever-increasing depth) but cheap insurance for the day
  aspiration windows are added.

---

## 7. Move ordering — guessing the best move first (`ordered_moves`)

Everything from §4 onward lives or dies on move order, so moves are sorted
into explicit tiers (highest first):

| Tier | Score | What |
|---|---|---|
| Hash move | 1,000,000 | The TT's stored best move for this position |
| Captures | 100,000 + MVV-LVA | Most Valuable Victim − Least Valuable Attacker |
| Killer 1, 2 | 90,000 / 80,000 | Quiet moves that recently caused cutoffs at this ply |
| Quiet moves | history value (≤ 70,000) | Ordered by past success anywhere in the tree |

* **Hash move**: even when a TT entry's *score* can't be used, its *move* is
  the single best guess available — it was literally proven best here before,
  usually by the previous iterative-deepening iteration. This is the main
  channel through which each iteration teaches the next.
* **MVV-LVA** (`10 * victim − attacker`): try QxR before PxP before RxP.
  Cheap and effective for ordering captures.
* **Killer moves** (2 slots per ply): sibling positions at the same ply are
  similar, so a quiet move that refuted one sibling (caused a beta cutoff)
  often refutes the others — e.g. the same knight-fork square. Stored per ply,
  most recent first.
* **History heuristic** (`history[from][to] += depth²` on quiet cutoffs): a
  global "this move keeps being good" table that orders the remaining quiet
  moves. Deeper cutoffs weigh more because they were more expensive to find.

Killers and history matter beyond their direct (modest) node savings: they are
what makes "late move = probably bad move" true enough for LMR (§8b) to be safe.

---

## 8. Searching *selectively* — spending depth where it matters

A fixed-depth search wastes most of its effort on hopeless moves. The next
three techniques bend the tree: promising/forcing lines get searched deeper,
unpromising ones shallower. Combined, they took the engine from ~depth 8 to
~depth 13–16 in the same 2 seconds.

### 8a. Principal Variation Search (PVS)

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

### 8b. Late Move Reductions (LMR)

Given good ordering, a *quiet* move that is 4th, 5th, 10th in the list is very
unlikely to be best. LMR searches such moves at **reduced depth** first
(`depth − 1 − r`, where `r` grows logarithmically with depth and move index).

Guards — never reduce: at `depth < 3`, the first 3 moves, captures/promotions,
when in check, or moves that give check (forcing/tactical moves get full
depth).

The safety net: if a reduced move *beats alpha anyway*, it is immediately
**re-searched at full depth** (and, if needed, once more with a full window —
the PVS machinery). So a wrongly-reduced move costs one extra search; it can
never sneak into the PV on the strength of a shallow score alone.

### 8c. Null-move pruning (verified)

The observation: in chess, having the move is almost always an advantage. So
ask: *"if I pass (do nothing) and my opponent moves twice, am I still doing
too well (≥ beta)?"* If yes, the real position — where I do get a move — is
surely a cutoff, so prune the whole node without searching my moves.
Implemented with `swap_turn()` and a reduced-depth (`R = 2–3`) null-window
search around beta.

**The flaw — zugzwang**: positions where every real move *loses* but passing
would be fine (trapped piece, king-and-pawn endgames). There "pass ≥ beta" says
nothing about real moves, and naive null-move prunes a lost position as won.
Zugzwang is a property of the whole move tree, so there is no cheap static
detector for it. Three defenses in this engine:

1. Disabled when the side to move has only king + pawns (where zugzwang is
   routine).
2. **No two null moves in a row** (`allow_null` flag) — a double pass is a
   degenerate non-position.
3. **Verification**: when the null search fails high, we do *not* prune
   immediately. We re-search the node at reduced depth with **real moves
   only** (null disabled). Genuine zugzwang → the verification fails low → we
   refuse to prune and search normally. The verification search is exactly the
   zugzwang detector that can't be written statically: it plays the moves.

Cost/benefit: roughly neutral in sharp tactical positions (verification
overhead ≈ savings), worth about +3 plies in quiet positions — which is where
most of a real game happens.

### 8d. Check extensions

The mirror image of reductions: when the side to move is **in check**, search
one ply *deeper* (`depth += 1`). Checks are forcing — few legal replies, and
tactics (mating attacks, perpetuals) hide behind them — so the extra depth
goes exactly where the tree is narrow and critical. It also guarantees we
never statically evaluate an in-check position (whose eval would be garbage —
quiescence only explores captures, not check evasions).

**The recursion trap**: +1 per check means depth never decreases along a
line of consecutive checks. Since the engine has no repetition detection yet,
a perpetual check would recurse forever and overflow the stack. The guard
`ply < MAX_PLY (128)` stops extending past that depth, bounding the recursion.
(Proper repetition detection is the "real" fix and a good future addition.)

---

## 9. How a move actually gets chosen — the full pipeline

Putting it all together, one engine move works like this:

```
best_move(pos)
└─ iterative deepening: for depth = 1, 2, 3, … until ~2s deadline
   └─ negamax(pos, depth, full window, ply=0)
      ├─ game over? → exact leaf score (mate/draw)
      ├─ in check? → depth += 1                       (check extension)
      ├─ depth == 0 → quiescence (captures only, stand-pat)
      ├─ TT probe → maybe instant cutoff (not at root); remember hash move
      ├─ null-move: pass, reduced search; fail-high → verify → maybe prune
      └─ for each move, ordered hash > MVV-LVA > killers > history:
         ├─ 1st move: full window, full depth
         ├─ later quiet/late moves: reduced depth (LMR), null window (PVS)
         │   └─ beats alpha? → re-search deeper / wider as needed
         ├─ beta cutoff? → record killer + history, store TT, stop loop
         └─ best move so far → extend the PV
      └─ store {depth, score, bound, best move} in TT
   └─ after each completed depth: progress callback → TUI shows
      "depth N, score, knodes/s" live (the PV is deliberately NOT shown —
      it would tell the human the engine's expected line)
└─ play pv[0]
```

---

## 10. Supporting cast

* **Progress reporting** (`SearchInfo`, `time_bound_search_with_progress`):
  because iterative deepening naturally pauses between depths, a plain
  synchronous callback after each finished depth gives smooth live updates
  with no threads and no races.
* **PGN export** (`play.rs::to_pgn`): the loop records every move in SAN
  (`SanPlus`, so `+`/`#` suffixes are right) and, at game end, emits a
  standard seven-tag PGN the TUI offers to save.
* **Tests**:
  * `tests/epd_tests.rs` — tactical positions (EPD files with a `bm` best
    move); the engine must find each within 1s. This is the regression suite
    that every search change was validated against.
  * `tests/nodes_bench.rs` (`#[ignore]`d) — "how deep in 2 seconds?"
    benchmark used to A/B each technique. Run with
    `cargo test --release --test nodes_bench -- --ignored --nocapture`.
  * Unit tests in `eval.rs` (symmetry, PST sanity) and `tui.rs` (cursor math).

### Measured impact (same hardware, ~2s per move)

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

## 11. What's deliberately *not* here (future work)

Roughly in order of value:

1. **Repetition & 50-move detection** — the engine can't yet recognize
   threefold repetition; it neither claims draws nor avoids them when winning.
   Also the proper fix for the perpetual-check recursion cap.
2. **Aspiration windows** — start each iteration with a narrow window around
   the previous score instead of (−∞, +∞); re-search wider on failure. (This
   is also what makes the root-TT-cutoff guard from §6 load-bearing.)
3. **Static Exchange Evaluation (SEE)** — evaluate capture sequences on one
   square without searching; used to skip losing captures in quiescence and
   order captures better than MVV-LVA.
4. **Richer evaluation** — mobility, passed pawns, king-safety terms, bishop
   pair.
5. **Opening book / endgame tablebases.**
6. **Multithreading (Lazy SMP)** — worth ~2–3× nodes; intentionally postponed
   because all of the above were worth more, single-threaded.
