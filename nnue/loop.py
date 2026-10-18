"""The self-play training loop: generate data, train, test, promote, repeat.

    python nnue/loop.py --run first --generations 10
    python nnue/loop.py --run first --report

Every generation g:

  1. datagen  the current best evaluator plays itself     -> data/gen_g.txt
  2. pack     text to training records                   -> data/gen_g.bin
  3. train    a new net on the last --window generations -> nets/gen_g.bin (+ .pt)
  4. check    engine and PyTorch agree on the new net (stops the loop if not)
  5. match    new net vs the current best                -> Elo
  6. match    new net vs PST, a fixed yardstick across generations
  7. promote  the new net becomes the best if it beat the best by > --promote-elo

Generation 1 starts from PST. Everything lives in runs/<run>/:

  log.txt      what happened, in order, readable
  log.jsonl    one JSON record per finished generation (what --report shows)
  logs/        full output of every step
  config.json  the settings, saved at the first start
  state.json   progress, saved after every step

The loop can be stopped at any time (Ctrl-C) and resumed with the same --run:
finished steps are skipped and an interrupted step starts over. Settings given
on the command line when resuming override the saved ones from then on.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release"
NNUE_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "generations": 10,
    "games": 100_000,  # datagen games per generation (~70 positions each)
    "datagen_nodes": 5000,
    "window": 4,  # train on the data of this many most recent generations
    "epochs": 15,
    "batch_size": 16384,
    "lr": 1e-3,
    "final_lr": 1e-5,
    "lambda": 0.75,  # 1.0 = train on search scores only, 0.0 = on game results only
    "init_from_best": False,  # start training from the best net's weights instead of from scratch
    "match_games": 400,
    "match_nodes": 10000,
    "pst_games": 400,  # games against PST each generation; 0 to skip
    "promote_elo": 0.0,
    "concurrency": 0,  # 0 = all cores
    "seed": 1,
    "delete_text": False,  # delete data/gen_g.txt once the generation is done (the .bin is kept)
}

STEPS = ["datagen", "pack", "train", "check", "match_best", "match_pst", "promote"]


class Run:
    def __init__(self, name):
        self.dir = ROOT / "runs" / name
        for sub in ["data", "nets", "logs"]:
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.config_file = self.dir / "config.json"

    def log(self, msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(line + "\n")

    def load_state(self):
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"generation": 1, "best": "pst", "best_generation": 0, "done": [], "record": {}}

    def save_state(self, state):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_file)

    def data(self, g, ext):
        return self.dir / "data" / f"gen_{g:03}.{ext}"

    def net(self, g, ext="bin"):
        return self.dir / "nets" / f"gen_{g:03}.{ext}"

    def logfile(self, g, step, ext="log"):
        return self.dir / "logs" / f"gen_{g:03}_{step}.{ext}"


def run_cmd(cmd, logfile):
    """Run a command, showing its output live and keeping a copy in `logfile`."""
    with open(logfile, "wb") as log:
        log.write(("$ " + " ".join(str(c) for c in cmd) + "\n").encode())
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            while chunk := proc.stdout.read1(4096):
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                log.write(chunk)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        if proc.wait() != 0:
            raise RuntimeError(f"{Path(str(cmd[0])).name} failed (exit {proc.returncode}), see {logfile}")
    print()


def duration(secs):
    secs = int(secs)
    return f"{secs // 3600}h{secs // 60 % 60:02}m" if secs >= 3600 else f"{secs // 60}m{secs % 60:02}s"


def elo_text(m):
    margin = "?" if m["margin"] is None else f"{m['margin']:.0f}"
    return f"{m['elo']:+.0f} ± {margin}  (W {m['wins']} D {m['draws']} L {m['losses']})"


def step_datagen(run, cfg, state, g):
    txt, partial = run.data(g, "txt"), run.data(g, "txt.partial")
    partial.unlink(missing_ok=True)
    summary = run.logfile(g, "datagen", "json")
    cmd = [BIN / "datagen", "--eval", state["best"], "--games", cfg["games"], "--nodes", cfg["datagen_nodes"],
           "--seed", cfg["seed"] * 1_000_003 + g, "--out", partial, "--json", summary]
    if cfg["concurrency"]:
        cmd += ["--concurrency", cfg["concurrency"]]
    run_cmd(cmd, run.logfile(g, "datagen"))
    partial.replace(txt)
    d = json.loads(summary.read_text())
    state["record"]["datagen"] = d
    n = max(1, d["games"])
    run.log(
        f"gen {g}  datagen ({state['best']}): {d['positions']:,} positions from {d['games']:,} games "
        f"in {duration(d['seconds'])}  (games: White {100 * d['white_wins'] / n:.0f}%, "
        f"draw {100 * d['draws'] / n:.0f}%, Black {100 * d['black_wins'] / n:.0f}%)"
    )


def step_pack(run, cfg, state, g):
    partial = run.data(g, "bin.partial")
    run_cmd([BIN / "pack", run.data(g, "txt"), partial], run.logfile(g, "pack"))
    partial.replace(run.data(g, "bin"))
    run.log(f"gen {g}  pack: {run.data(g, 'bin').stat().st_size // 68:,} records")


def step_train(run, cfg, state, g):
    window = [run.data(h, "bin") for h in range(max(1, g - cfg["window"] + 1), g + 1)]
    log = run.logfile(g, "train", "jsonl")
    log.unlink(missing_ok=True)
    cmd = [sys.executable, NNUE_DIR / "train.py", *window, "--out", run.net(g), "--epochs", cfg["epochs"],
           "--batch-size", cfg["batch_size"], "--lr", cfg["lr"], "--final-lr", cfg["final_lr"],
           "--lambda", cfg["lambda"], "--log", log, "--seed", cfg["seed"] + g]
    if cfg["init_from_best"] and state["best_generation"] > 0:
        cmd += ["--init", run.net(state["best_generation"], "pt")]
    run_cmd(cmd, run.logfile(g, "train"))
    epochs = [json.loads(line) for line in log.read_text().splitlines()]
    state["record"]["train"] = {"data_generations": [p.stem for p in window], "epochs": epochs}
    last = epochs[-1]
    run.log(
        f"gen {g}  train: {len(epochs)} epochs on {len(window)} generation(s) of data, "
        f"loss train {last['train_loss']:.5f} val {last['val_loss']:.5f}"
    )


def step_check(run, cfg, state, g):
    summary = run.logfile(g, "check", "json")
    try:
        run_cmd(
            [sys.executable, NNUE_DIR / "check.py", run.net(g, "pt"), run.net(g), run.data(g, "txt"), "--json", summary],
            run.logfile(g, "check"),
        )
    finally:
        if summary.exists():
            c = json.loads(summary.read_text())
            state["record"]["check"] = c
            run.log(f"gen {g}  check: engine vs PyTorch mean diff {c['mean_diff']:.2f} cp, max {c['max_diff']:.1f} cp"
                    + ("" if c["ok"] else "  MISMATCH"))


def match(run, cfg, g, opponent, games, name):
    summary = run.logfile(g, name, "json")
    cmd = [BIN / "arena", f"nnue:{run.net(g)}", opponent, "--games", games, "--nodes", cfg["match_nodes"],
           "--seed", cfg["seed"] * 7919 + g, "--json", summary]
    if cfg["concurrency"]:
        cmd += ["--concurrency", cfg["concurrency"]]
    run_cmd(cmd, run.logfile(g, name))
    return json.loads(summary.read_text())


def step_match_best(run, cfg, state, g):
    m = match(run, cfg, g, state["best"], cfg["match_games"], "match_best")
    state["record"]["match_best"] = {**m, "opponent": best_name(state)}
    run.log(f"gen {g}  vs best ({best_name(state)}): {elo_text(m)}")


def step_match_pst(run, cfg, state, g):
    # Against PST the first match already is this one.
    if state["best"] == "pst":
        state["record"]["match_pst"] = state["record"]["match_best"]
        return
    if cfg["pst_games"] <= 0:
        return
    m = match(run, cfg, g, "pst", cfg["pst_games"], "match_pst")
    state["record"]["match_pst"] = m
    run.log(f"gen {g}  vs pst: {elo_text(m)}")


def step_promote(run, cfg, state, g):
    m = state["record"]["match_best"]
    promoted = m["elo"] > cfg["promote_elo"]
    state["record"]["promoted"] = promoted
    if promoted:
        state["best"], state["best_generation"] = f"nnue:{run.net(g)}", g
        run.log(f"gen {g}  promoted: gen {g} is the new best ({m['elo']:+.0f} Elo > {cfg['promote_elo']:+.0f})")
    else:
        run.log(f"gen {g}  not promoted: best stays {best_name(state)} ({m['elo']:+.0f} Elo <= {cfg['promote_elo']:+.0f})")


def best_name(state):
    return "pst" if state["best"] == "pst" else f"gen {state['best_generation']}"


STEP_FNS = {
    "datagen": step_datagen,
    "pack": step_pack,
    "train": step_train,
    "check": step_check,
    "match_best": step_match_best,
    "match_pst": step_match_pst,
    "promote": step_promote,
}


def report(run):
    path = run.dir / "log.jsonl"
    if not path.exists():
        print("no finished generations yet")
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print(f"{'gen':>3}  {'data from':<9} {'positions':>11}  {'val loss':>8}  {'check':>6}  "
          f"{'vs best':>17}  {'vs pst':>17}  {'promoted':<8}  {'time':>7}")
    for r in rows:
        last = r["train"]["epochs"][-1]
        def e(m):
            if not m:
                return "-"
            margin = "?" if m["margin"] is None else f"{m['margin']:.0f}"
            return f"{m['elo']:+.0f} ± {margin}"
        print(f"{r['generation']:>3}  {r['best_before']:<9} {r['datagen']['positions']:>11,}  {last['val_loss']:>8.5f}  "
              f"{r['check']['mean_diff']:>5.1f}c  {e(r.get('match_best')):>17}  {e(r.get('match_pst')):>17}  "
              f"{'yes' if r['promoted'] else 'no':<8}  {duration(r['seconds']):>7}")
    state = run.load_state()
    print(f"\nbest: {best_name(state)}" + ("" if state["best"] == "pst" else f"  ({state['best'].removeprefix('nnue:')})"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="name of the run; its files live in runs/<name>/")
    p.add_argument("--report", action="store_true", help="print a summary of the run and exit")
    for key, default in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(flag, type=type(default), default=None, help=f"default {default}")
    args = p.parse_args()
    run = Run(args.run)

    if args.report:
        report(run)
        return

    # Saved settings, overridden by anything given on this command line.
    cfg = dict(DEFAULTS)
    if run.config_file.exists():
        cfg.update(json.loads(run.config_file.read_text()))
    overrides = {k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}
    changed = {k: v for k, v in overrides.items() if cfg.get(k) != v}
    cfg.update(overrides)
    run.config_file.write_text(json.dumps(cfg, indent=2))

    state = run.load_state()
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    run.log(f"start: run {args.run}, generation {state['generation']} of {cfg['generations']}, best {best_name(state)}, "
            f"engine commit {commit or 'unknown'}" + (f", changed settings {changed}" if changed else ""))

    run.log("building engine binaries")
    run_cmd(["cargo", "build", "--release", "--bin", "datagen", "--bin", "pack", "--bin", "arena", "--bin", "evalfen"],
            run.dir / "logs" / "build.log")

    try:
        while state["generation"] <= cfg["generations"]:
            g = state["generation"]
            if not state["done"]:
                state["record"] = {"generation": g, "best_before": best_name(state), "started": time.time(), "steps": {}}
                run.log(f"gen {g}  start, best is {best_name(state)}")
            for step in STEPS:
                if step in state["done"]:
                    continue
                t0 = time.time()
                STEP_FNS[step](run, cfg, state, g)
                state["record"]["steps"][step] = time.time() - t0
                state["done"].append(step)
                run.save_state(state)

            record = state["record"]
            record["seconds"] = time.time() - record["started"]
            record["best_after"] = best_name(state)
            with open(run.dir / "log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            if cfg["delete_text"]:
                run.data(g, "txt").unlink(missing_ok=True)
            run.log(f"gen {g}  done in {duration(record['seconds'])}")
            state.update({"generation": g + 1, "done": [], "record": {}})
            run.save_state(state)
    except KeyboardInterrupt:
        run.save_state(state)
        run.log(f"interrupted in generation {state['generation']}; resume with the same --run")
        sys.exit(130)
    except Exception as e:
        run.save_state(state)
        run.log(f"stopped in generation {state['generation']}: {e}")
        sys.exit(1)

    run.log(f"finished {cfg['generations']} generations, best is {best_name(state)}")
    report(run)


if __name__ == "__main__":
    main()
