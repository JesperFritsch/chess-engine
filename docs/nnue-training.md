# NNUE: Evaluators, Self-Play and Training

How the engine learns its own evaluation: a neural network (an **NNUE**)
trained on positions from games the engine plays against itself. This document
covers the code that makes that possible, the file formats in between, and how
to run training. It assumes the search described in
[how-the-engine-works.md](how-the-engine-works.md).

---

## 1. The big picture

The hand-written evaluation (`eval.rs`: material plus piece-square tables) is
replaced by a small neural network that predicts how good a position is. The
network learns from examples (*this position → this value*), and the
examples come from the engine itself:

```
           ┌──────────────────────────────────────────────────────────────────┐
           │                                                                  │
           ▼                                                                  │
 best evaluator ──► datagen ──► data.txt ──► pack ──► data.bin ──► train.py ──► net.bin
 (PST at first)    self-play    fen|score|   (Rust)   68-byte      (PyTorch,    quantized
                   games        result                records      GPU/CPU)     weights
                                                                                  │
                         arena: new net vs best, new net vs PST ◄─────────────────┘
                         promoted to "best" if it wins
```

Each turn of this circle is a **generation**. A better evaluator plays better
games, better games make better training data, and better data trains a
better net. `nnue/loop.py` runs the circle unattended and logs every step.

What was built, by layer:

| Layer | Files | Role |
|---|---|---|
| Evaluator interface | `src/engine/evaluator.rs` | `Evaluator` trait; `PstEval`, `MaterialEval` |
| NNUE inference | `src/engine/nnue.rs` | `Network`, `NnueEval` with incremental accumulator |
| Search | `src/engine/search.rs` | generic over the evaluator, calls its move hooks |
| Games | `src/arena.rs` | play games, openings, adjudication, parallel runner |
| Binaries | `src/bin/{arena,datagen,pack,evalfen}.rs` | match, data generation, packing, static eval |
| Training | `nnue/train.py`, `nnue/check.py`, `nnue/loop.py` | trainer, engine/PyTorch check, the loop |
| Benchmarks | `tests/eval_bench.rs`, `tests/nodes_bench.rs` | speed and node counts per evaluator |

---

## 2. The `Evaluator` trait (`engine/evaluator.rs`)

The search no longer calls `eval::evaluate` directly. It calls an
**evaluator**, any type implementing:

```rust
pub trait Evaluator: Clone + Send + Sync + 'static {
    fn reset(&mut self, pos: &Chess, ply: usize);
    fn on_move(&mut self, ply: usize, parent: &Chess, mv: Move);
    fn on_null_move(&mut self, ply: usize) {}
    fn evaluate(&mut self, pos: &Chess, ply: usize) -> i32;
}
```

`evaluate` returns centipawns from the side to move's point of view, as
before. The other three methods are **hooks**: the search tells the evaluator
about every move it makes.

**Why hooks?** A stateless evaluator like the PST one doesn't need them; its
implementations are empty. The NNUE does. Its first layer is a sum of one
weight row per piece on the board, about 30 rows of 256 numbers. Recomputing
that for every evaluation is expensive, but a move only changes 2–4 pieces, so
the child's sum can be derived from the parent's by subtracting and adding a
few rows. To do that the evaluator must know *which move* led from *which
parent*, which a bare `evaluate(pos)` can't tell it.

**Why `ply` everywhere?** The evaluator keeps one state per ply. `on_move(ply,
…)` computes the state for `ply + 1` from the state at `ply`. There's no undo
step: the next sibling move simply overwrites `ply + 1` again.

**Ownership.** Every search thread owns its own evaluator (it lives in
`Worker`). A helper thread joining a split point clones a pristine copy (the
engine's `proto`) and calls `reset` at the split position's ply, because it
has no parent state of its own.

**The search is generic.** `SearchEngine<E: Evaluator = PstEval>`. The
compiler builds one copy of the search per evaluator and can inline the eval
call. That's faster than `dyn Evaluator`, which would go through a function
pointer millions of times per second. `SearchEngine::new(...)` still means the
PST eval, so the UCI engine, the TUI and existing tests are unchanged;
`SearchEngine::with_evaluator(e, tt_mb, threads)` selects another one.

Implementations:

* `PstEval`: the existing `eval::evaluate`.
* `MaterialEval`: material only, deliberately weak. It's a baseline and a
  sanity check for the comparison tools.
* `NnueEval`: the network (§3).

Where the search calls the hooks: `on_move` in `search_move` and in
`quiescence` (which now takes a `ply` parameter), `on_null_move` before the
null-move search, `reset` at the start of a search and when a helper joins a
split point.

---

## 3. The network and `NnueEval` (`engine/nnue.rs`)

### Architecture: `(768 → 256) × 2 → 1`

```
 white view: 768 inputs ──► 256 accumulator ┐
                                            ├─► [side to move | other side] ─► CReLU ─► 1 output ─► ×400 = cp
 black view: 768 inputs ──► 256 accumulator ┘     (512 values)              clamp 0..1
             (same weights for both views)
```

* **Inputs.** One input per (color, piece type, square): 2 × 6 × 64 = 768,
  each 0 or 1. A position switches on one per piece, about 30.
  Feature index = `color * 384 + role * 64 + square`, with role 0..5 =
  pawn..king, square 0..63 = a1..h8, and color 0 = the viewing side's pieces
  (in White's view that's White, in Black's view Black).
* **Two views.** White's view is the board as it stands. Black's view is the
  board **flipped vertically with colors swapped**, so a feature always means
  "my pawn on e2" rather than "White's pawn on e2". Both views use the same
  weights; the net only has to learn each pattern once.
* **Side to move first.** The two accumulators are concatenated with the side
  to move's first. That puts the side-to-move perspective in the
  architecture itself, which is what negamax wants.
* **CReLU** clamps each value to 0..1. The output is a single number in units
  of 400 cp.

### Quantization: integer math in the engine

The trainer works in floats; the engine works in integers, which are faster
and exactly reproducible. On export the weights are scaled and rounded:

| Part | Scale | Type |
|---|---|---|
| first-layer weights and bias | ×255 (`QA`) | i16 |
| output weights | ×64 (`QB`) | i16 |
| output bias | ×255×64 | i32 |

In the engine, CReLU becomes `clamp(acc, 0, 255)`, and the evaluation is
`(Σ clamp(acc)·w + bias) × 400 / (255 × 64)`. During training every weight is
clipped to ±1.98 after each step. That guarantees the i16 accumulator
(bias + ~32 rows × 505) can never overflow.

### The incremental accumulator

`NnueEval` holds an `Arc<Network>` (read-only weights, shared by every thread)
and a **stack of accumulators, one per ply** (each a pair: White's and Black's
view).

* `reset` builds `stack[ply]` from scratch: bias plus one row per piece.
* `on_move` copies `stack[ply]` to `stack[ply + 1]`, then subtracts the rows of
  pieces that left a square and adds the rows of pieces that arrived:
  * normal move: −piece@from, +piece@to (the promoted piece, for a promotion);
  * capture: additionally −captured@to;
  * en passant: −pawn@from, +pawn@to, −captured pawn beside `from`;
  * castling: shakmaty encodes it as "king takes own rook"; −king, −rook,
    +king@g/c-file, +rook@f/d-file.
* `on_null_move` only copies: nothing moves, and the side to move is handled
  in `evaluate`.
* The stack is allocated on the first `reset`, so cloning a fresh evaluator for
  a helper thread is cheap.

**Self-check in debug builds.** `evaluate` asserts that the incrementally
updated accumulator equals one built from scratch. Any mistake in the move
updates fails immediately in a debug build. It was verified over 535,643
nodes on the bench positions, which include castling, en passant and
promotions. Release builds skip the check.

**Speed:** about 3.0M nodes/s, against 3.5M for PST, on the bench (one
thread, depth 10).

### The net file

Written by `export()` in `train.py` and read by `Network::from_bytes`, all
little-endian:

```
"NNUE"             4 bytes magic
version            u32 = 1
hidden size        u32 = 256
ft weights         i16 [768][256]
ft bias            i16 [256]
out weights        i16 [512]     side to move's half first
out bias           i32
```

394,768 bytes for 256 hidden. The loader checks the magic, version, hidden
size and length, and gives a clear error for a wrong or stale file.

**The constants must match** between `train.py` and `nnue.rs`: `HIDDEN`, `QA`,
`QB`, `SCALE` and the feature indexing. `check.py` (§6) catches a mismatch.

---

## 4. Playing games (`src/arena.rs`)

A library module with nothing match- or datagen-specific in it. Both binaries
build on it.

* **`play_game(white, black, opening, config) -> GameRecord`** plays one game
  between two `&mut dyn ChessEngine` (any engine, any evaluator). Both
  engines are cleared first. The record holds, for every searched move, the
  position, the move, the score, depth and nodes, plus the result and the
  `Termination`.
* **Game end** follows the rules (checkmate, stalemate, insufficient material,
  fifty-move rule, threefold repetition) plus **adjudication**
  (`Adjudication`, defaults):
  * win once |score| ≥ 1000 cp with the same side ahead for 4 plies in a row;
  * draw once |score| ≤ 10 cp for 8 plies in a row, from ply 80 on;
  * draw at 400 plies.

  Both engines' scores count (they alternate), so both must agree. In practice
  most games end by adjudication, which saves a lot of time.
* **`run_games(jobs, concurrency, config, make_engines, make_opening, on_game)`**
  plays many games in parallel. Each thread calls `make_engines` once and
  reuses its pair of engines. Openings are built **on the worker thread** by
  `make_opening(rng, engine)`, because they may need searches. Each `Job`
  carries a seed; equal seeds give equal openings. Finished games go to
  `on_game` on the calling thread, so it can write files without locking.
* **`make_engine(name, hash_mb)`** maps an evaluator name to a single-threaded
  engine: `pst`, `material` or `nnue:<path>`. It returns an error message for
  an unknown name or an unreadable net, and is shared by all binaries.

### Openings

Self-play at a fixed node count is **deterministic**: the same opening always
gives the same game. Openings therefore have to supply all the variety.

* **`random_opening(rng, plies)`**: N random moves. Used by the arena, where
  both games of a pair must get the same opening whatever the engines are.
* **`mixed_opening(rng, engine, limits, &OpeningConfig)`**: used by datagen.
  Every 3rd ply is random and the others are the engine's (plies 1–2 engine,
  ply 3 random, …). The random moves give variety; the engine moves in between
  keep positions realistic. Defaults (`OpeningConfig`):
  * `random_every: 3`. An odd period makes the random plies alternate between
    White and Black, each answered by the opponent's engine move. With period
    2, one color would always be the random one.
  * `random_plies: 8`, **or 9, chosen per game.** The side that moves right
    after the *last* random ply is the first to exploit it, and measurably
    wins more often: with 8, White won 25k positions against Black's 19k; with
    7 the bias flipped. Choosing 8 or 9 at random gives that move to each
    color half the time.
  * `max_score: 400`. After the opening the engine searches once; openings it
    scores beyond ±4 pawns are discarded and rebuilt. A game decided by a
    random blunder teaches little.

  With all three: 53.6k / 57.4k White/Black-win positions over 2,000 games, and
  a symmetric score distribution.

---

## 5. The binaries

All are built by `cargo build --release` and land in `target/release/`.

### `arena`: evaluator vs evaluator

```
cargo run --release --bin arena -- pst material --games 200 --nodes 25000
cargo run --release --bin arena -- nnue:nets/gen_003.bin nnue:nets/gen_002.bin
```

Options: `--games` (200), `--nodes` per move (25000), `--concurrency` (all
cores), `--opening-plies` (8), `--seed` (1), `--hash` MB (16), `--json FILE`
(machine-readable summary).

* Each random opening is played **twice, colors swapped**, so a lopsided
  opening helps both sides equally.
* It reports W/D/L from the first evaluator's view, the **Elo difference with
  a 95% margin**, and how games ended.
* The margin is computed **per opening pair** (pentanomial). The two games of a
  pair share an opening and aren't independent, so counting games separately
  would understate the error.
* It's deterministic: the same seed gives the same result. Swapping the two
  evaluators flips the sign exactly.

Sanity result: PST beats material-only by about +250 Elo.

### `datagen`: self-play training data

```
cargo run --release --bin datagen -- --eval pst --games 10000 --nodes 5000 --out data.txt
```

Options: `--eval` (pst), `--games` (1000), `--nodes` per move (5000),
`--concurrency`, `--random-every` (3), `--random-plies` (8),
`--max-opening-score` (400), `--seed` (current time, so runs differ),
`--hash`, `--out` (data.txt), `--json FILE`.

The engine plays itself (two engines with the same evaluator) from mixed
openings. Every searched position after the opening that passes the filters
becomes one line:

```
r1bqkbnr/1p2pppp/p1n5/8/3P4/6P1/PP1PNP1P/RNBQKB1R w KQkq - 1 7 | 80 | 0.0
<FEN>                                                           score  result
```

* **score**: the search score in centipawns, **White's point of view**.
* **result**: the game's outcome for White: 1.0 win, 0.5 draw, 0.0 loss.

The FEN says who is to move, so the trainer can flip to the side to move.

**Filters** (`keep()`): positions where the side to move is in check, where
the played move is a capture or promotion, or with a mate score. The net
should learn the value of a *quiet* position. In those cases the search score
describes what happens after the tactic resolves, not the position itself.
About 70% of positions are kept.

The file is **appended to and flushed after every game**. Ctrl-C loses only
the games in progress, and several runs can fill one file.

Throughput on the 14-core laptop at 5,000 nodes per move: about 2,400
positions/s, roughly 8.7M per hour.

### `pack`: text to training records

```
cargo run --release --bin pack -- data.txt data.bin
```

Python is slow at parsing FENs, so this happens once, in Rust. Each line
becomes a fixed **68-byte record** (little-endian):

| Field | Type | Meaning |
|---|---|---|
| features | u16 × 32 | piece features from White's view (§3); unused slots hold 768 |
| score | i16 | centipawns, White's view |
| result | u8 | 0 Black won, 1 draw, 2 White won |
| stm | u8 | side to move: 0 White, 1 Black |

The file describes only the board as it stands. Black's view and the flip to
the side to move happen in the trainer. 100M positions ≈ 6.8 GB.

### `evalfen`: static evaluation

```
echo "<fen>" | target/release/evalfen nnue:nets/net.bin
```

Prints the evaluator's static eval (cp, side to move) for each FEN on stdin.
Used by `check.py`.

---

## 6. Training (`nnue/`)

### `train.py`: the trainer

```
python nnue/train.py data.bin [more.bin ...] --out nets/net.bin --epochs 10
```

Uses CUDA when available, otherwise the CPU. Options: `--epochs` (10),
`--batch-size` (16384), `--lr` (1e-3), `--final-lr` (1e-5), `--lambda` (0.75),
`--val-fraction` (0.01), `--resume` / `--init` checkpoint, `--log FILE`,
`--seed`.

What happens, in order:

1. **Data.** The record files are memory-mapped, so they don't have to fit in
   RAM, and used together as one dataset. The **last 1% of each file** is held
   out for validation. Positions from one game sit next to each other, so a
   random split would validate on near-copies of training positions and look
   better than it is.
2. **Each batch:** compute Black's view from White's (`flip`: swap colors,
   mirror ranks), put the side to move first, and flip score and result when
   Black is to move.
3. **Target** = `λ · sigmoid(score / 400) + (1 − λ) · result`. λ = 1 trains
   only on search scores, λ = 0 only on game results. The score is precise but
   limited by the search that produced it; the result is ground truth but very
   noisy for any single position.
4. **The net.** The first layer is an `EmbeddingBag` in sum mode. It adds the
   weight rows of the pieces present, which is exactly a linear layer over the
   0/1 inputs, but without multiplying 738 zeros.
5. **Loss** = mean squared error between `sigmoid(output)` and the target.
   The optimizer is **Adam**, with the learning rate decaying from 1e-3 to
   1e-5 along a cosine curve.
6. **Clip** all weights to ±1.98 after every step (quantization safety, §3).
7. **After every epoch:** print train and validation loss, save a checkpoint
   (`net.pt`, for `--resume`), export the quantized `net.bin`, and append a JSON
   line to `--log`.

`--resume` continues an interrupted run (weights, optimizer, schedule).
`--init` starts a new run from an earlier net's weights with a fresh
optimizer, i.e. fine-tuning.

**Reading the losses.** Both should fall. Validation close to training means
the net generalizes. Training falling while validation rises means it's
memorizing; it needs more data or fewer epochs. The absolute value depends on
the data and λ, so compare runs on the same data.

**Speed.** On the laptop CPU, batch preparation (numpy) runs at about 1.3M
positions/s and the training step at about 0.17M/s. On the GPU the step gets
much faster and batch preparation becomes the limit, so expect about 1M+
positions/s, i.e. an epoch over 100M positions in roughly 1.5 minutes. Moving
`flip` and friends onto the GPU would lift that limit if it becomes a
problem.

### `check.py`: engine vs PyTorch

```
python nnue/check.py nets/net.pt nets/net.bin data.txt
```

Evaluates 2,000 positions three ways: with the float net in PyTorch (from the
checkpoint), with the exported net read back and **dequantized**
(`train.load_exported`: the integers in the `.bin` divided by `QA`/`QB`, i.e.
the weights the engine actually holds, as floats), and with the engine itself
(`evalfen`). That gives two differences, which answer two different questions:

* **engine vs dequantized** is the correctness test, and what the script passes
  or fails on. Both sides hold identical weights, so they can differ only by
  the engine's final integer division: about 0.5 cp, never more than 1.
  Anything larger means the two implementations disagree about the network
  itself (a feature indexed differently, a view flipped the wrong way, a
  constant out of step), and the engine would be playing a different net than
  the one trained. It fails (exit code 1) above a mean of 1.5 cp
  (`--max-mean-diff`).
* **float vs dequantized** is the **quantization loss**: what rounding the
  weights to integers costs. Reported for information only; it is not a bug and
  has no correct value. It scales with the size of the evaluations, so it grows
  as the net gets stronger — about 2 cp on a freshly initialized net, 8 cp on
  one trained to an average |eval| of 400 cp.

Comparing float against the engine directly, as this script first did, measures
both at once and can do neither job. On a weak net every evaluation is near
zero, so *any* two networks agree to within a couple of centipawns and a real
mismatch passes unnoticed; on a trained net the honest quantization loss trips
a threshold meant for implementation bugs. Measured on generation 1 of the
first run: engine vs dequantized 0.49 cp, while deliberately swapping the
output halves, leaving Black's view unflipped, or dropping the colour offset
from the feature index land at 834, 210 and 4,961 cp respectively.

### `loop.py`: the training loop

```
python nnue/loop.py --run first --generations 10
python nnue/loop.py --run first --report
```

Every generation `g`:

| Step | What | Output |
|---|---|---|
| 1. datagen | the current best evaluator plays itself (PST in gen 1) | `data/gen_g.txt` |
| 2. pack | text to records | `data/gen_g.bin` |
| 3. train | a new net on the last `--window` generations' data | `nets/gen_g.bin`, `.pt` |
| 4. check | engine vs PyTorch; **stops the loop on a mismatch** | |
| 5. match_best | new net vs the current best | Elo |
| 6. match_pst | new net vs PST, a fixed yardstick across generations | Elo |
| 7. promote | new net becomes best if it beat best by > `--promote-elo` | |

In generation 1 the best *is* PST, so step 6 reuses step 5's result.

**Everything lives in `runs/<name>/`** (ignored by git):

* `log.txt`: a readable timeline. The start line records the engine's git
  commit and any changed settings; then one line per step, e.g.
  `gen 3  vs best (gen 2): +24 ± 31  (W 140 D 95 L 165)`.
* `log.jsonl`: one JSON record per finished generation with every number:
  datagen stats, loss per epoch, check result, both matches, time per step.
* `logs/gen_g_<step>.log`: the full output of each step (plus `.json` summaries).
* `data/`, `nets/`: the data and nets of every generation.
* `config.json`: settings, saved at the first start.
* `state.json`: progress, saved after every step.

**Stopping and resuming.** Ctrl-C (or a crash) at any time; run the same
command again to continue. Finished steps are skipped and an interrupted step
starts over. datagen and pack write to `.partial` files, so half-written
data is never used. Settings given on the command line when resuming override
the saved ones from then on, and the change is noted in `log.txt`.

**Settings** (`--help` lists them all):

| Option | Default | Meaning |
|---|---|---|
| `--generations` | 10 | stop after this generation |
| `--games` | 100000 | datagen games per generation (~70 positions each) |
| `--datagen-nodes` | 5000 | nodes per move in datagen |
| `--window` | 4 | train on the data of this many most recent generations |
| `--epochs` | 15 | |
| `--batch-size` | 16384 | |
| `--lr`, `--final-lr` | 1e-3, 1e-5 | cosine schedule |
| `--lambda` | 0.75 | search score vs game result in the target |
| `--init-from-best` | off | fine-tune from the best net instead of training from scratch |
| `--match-games` | 400 | games vs the best |
| `--match-nodes` | 10000 | nodes per move in matches |
| `--pst-games` | 400 | games vs PST; 0 skips |
| `--promote-elo` | 0 | promotion threshold |
| `--concurrency` | 0 (all cores) | threads for datagen and matches |
| `--seed` | 1 | base seed; datagen, training and matches derive theirs from it |
| `--delete-text` | off | delete `gen_g.txt` after its generation (the `.bin` is kept) |

---

## 7. Running it on the GPU machine

### Setup

```sh
git pull                                   # this code must be committed and pushed first
cargo build --release

uv venv --python 3.12 nnue/.venv
VIRTUAL_ENV=nnue/.venv uv pip install --index-url https://download.pytorch.org/whl/cu128 torch numpy
nnue/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The RTX 5070 Ti is a Blackwell card and needs a CUDA 12.8+ build of PyTorch
(the `cu128` index above, or newer). The last line must print `True`;
otherwise training silently runs on the CPU (`train.py` prints `device cpu`
at the start).

### Smoke test first

A few minutes, to see every step work together before committing hours:

```sh
nnue/.venv/bin/python nnue/loop.py --run smoke --generations 2 \
    --games 300 --epochs 2 --match-games 20 --pst-games 20
nnue/.venv/bin/python nnue/loop.py --run smoke --report
```

Expect the check step to report about 0.5 cp engine vs dequantized, and the
nets to be weak (300 games is very little data) — they will lose every game to
PST, which at two gradient steps is correct behaviour, not a failure. The point
is that nothing crashes. Delete `runs/smoke/` afterwards.

### The real run

```sh
nnue/.venv/bin/python nnue/loop.py --run first --generations 10
```

Rough time per generation with the defaults (100k games ≈ 7M positions): data
generation dominates, at 50 minutes on 14 laptop cores and less on a faster
CPU. Training on 4 generations × 7M positions × 15 epochs is a few minutes on
the GPU; the matches take a few minutes.

To keep it running after closing the terminal, start it in `tmux`/`screen`,
or with `nohup … &`. Everything important is in `runs/first/log.txt` anyway.

### Reading the results

```
nnue/.venv/bin/python nnue/loop.py --run first --report
```

```
gen  data from   positions  val loss   check            vs best             vs pst  promoted     time
  1  pst         7,012,345   0.03512    1.2c        +85 ± 30          +85 ± 30  yes        1h02m
  2  gen 1       7,100,212   0.03301    1.1c        +40 ± 28         +128 ± 31  yes        1h01m
  ...
```

(Illustrative numbers.) What to look for:

* **vs pst** is the progress curve. It should rise over the generations, by
  smaller amounts each time.
* **vs best** is noisy: ±30 Elo at 400 games. A single non-promotion means
  little; several in a row mean the loop has plateaued. Then the usual levers
  are more data per generation (`--games`), a larger `--window`, more
  datagen nodes (better labels), or `--lambda`.
* **check** should stay at a few cp. The loop stops by itself if it doesn't.
* **val loss** should drop as data grows. Values from different data aren't
  strictly comparable, since each generation's data comes from a different
  engine.

### Running pieces by hand

```sh
# compare any two nets, or a net against PST
target/release/arena nnue:runs/first/nets/gen_004.bin pst --games 1000 --nodes 10000

# speed and node counts of a net on the bench positions
NNUE_NET=runs/first/nets/gen_004.bin cargo test --release --test eval_bench -- --ignored --nocapture

# retrain on existing data with other settings
nnue/.venv/bin/python nnue/train.py runs/first/data/gen_00{1,2,3,4}.bin --out nets/experiment.bin --lambda 0.5
```

---

## 8. Other changes made along the way

* **Search speed-ups** (`search.rs`):
  * legal moves are generated once per node (the separate `is_game_over` call
    is gone; no legal moves now means mate or stalemate, checked where moves
    are generated);
  * quiescence generates only captures and promotions;
  * the Zobrist key is updated incrementally (`update_zobrist_hash`) and
    passed down the search, including to split points;
  * leaf nodes are no longer counted twice.
* **Release profile** in `Cargo.toml`: `lto = "fat"`, `codegen-units = 1`,
  `panic = "abort"`.
* `Score` derives `Clone, Copy, Debug`.
* `tests/eval_bench.rs`: compares evaluators on 12 fixed positions at a fixed
  depth (deterministic node counts, nps, chosen move and score per position).
  `NNUE_NET=…` adds a net, `BENCH_DEPTH=…` changes the depth.

---

## 9. Playing with a net (`AnyEval`)

`arena` and `datagen` pick their evaluator by name, but they do it once, at
startup, and build a `SearchEngine<PstEval>` or a `SearchEngine<NnueEval>`
accordingly. The UCI engine cannot: a GUI sets `EvalFile` whenever it likes,
and `SearchEngine<E>` fixes `E` when the binary is compiled.

`AnyEval` (`engine/evaluator.rs`) is the answer -- an enum over the three
evaluators that is itself an `Evaluator`, dispatching with a `match` in each
hook. That costs one branch per call rather than the indirect call a
`dyn Evaluator` would need, and it is confined to the interactive binaries:
`make_engine` still builds `SearchEngine` over the *concrete* type, so self-play
and matches search exactly as they always have.

Two spellings parse an evaluator:

* `AnyEval::parse` is strict -- `pst`, `material` or `nnue:<path>` -- and is
  what `make_engine` uses, where a typo should be an error.
* `AnyEval::parse_setting` is forgiving, for a value a person or a GUI typed:
  empty (or the literal `<empty>` some GUIs send) and `none` mean PST, and
  anything that is not a known name is taken to be a path to a net.

### The UCI option

```
setoption name EvalFile value runs/first/nets/gen_007.bin
setoption name EvalFile value nnue:nets/net.bin   # the explicit spelling
setoption name EvalFile value material            # the other evaluators
setoption name EvalFile value                     # back to PST
```

The net is read on the UCI thread, on the way in, so a bad path is answered
with `info string EvalFile: <the error>` and **the evaluator in use is left
alone** -- a GUI that mistypes a path keeps playing with the previous one
instead of the engine dying mid-game. A successful change reports
`info string EvalFile: now using nnue`, and clears the transposition table,
because every score in it came from the evaluator being replaced.

For lichess-bot, in `config.yml`:

```yaml
engine:
  uci_options:
    EvalFile: /path/to/chess-engine/runs/first/nets/gen_007.bin
    Threads: 4
    Hash: 256
```

### On the command line

Both interactive binaries take an evaluator as their first argument, which only
sets the one to *start* with (`EvalFile` can still replace it later):

```sh
target/release/uci-engine runs/first/nets/gen_007.bin
target/release/chess-engine runs/first/nets/gen_007.bin   # the TUI
target/release/chess-engine material
```

A name that is neither known nor a readable net exits with status 2 and the
usage line. The TUI checks it *before* taking over the terminal, so the error
is readable.

---

## 10. Not done yet

* **A net is not embedded in the binary.** The UCI engine and the TUI can
  both play with one (§10), but the file has to be there to be loaded.
  `include_bytes!` would make a single self-contained binary, which is what a
  release wants.
* **Batch preparation in `train.py` runs in numpy on the CPU.** On a fast GPU it
  will be the bottleneck. It can be moved onto the GPU if needed.
* **Bigger nets and more inputs.** King-bucketed inputs (HalfKP/HalfKA),
  output buckets by material, a second hidden layer, SIMD intrinsics for the
  accumulator. All are possible later; the trait and file format would need a
  version bump.
* **Mate display bug:** `Score::Mate` prints a negative mate as `-#-3`.
