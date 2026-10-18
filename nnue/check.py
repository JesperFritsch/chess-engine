"""Check that the engine evaluates a net the way PyTorch does.

    python nnue/check.py nets/net.pt nets/net.bin data.txt

Takes positions from a datagen file, evaluates them with the float network in
PyTorch (from the checkpoint) and with the engine's integer implementation (via
the `evalfen` binary on the exported net), and compares the two.

They can never agree exactly: the engine uses weights rounded to integers. A
few centipawns of difference is that rounding. Tens or hundreds mean the two
implementations disagree about the network itself (a feature indexed
differently, a view flipped the wrong way, a constant out of step), and the
engine is then playing with a different net than the one that was trained.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import train

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="the .pt checkpoint written by train.py")
    p.add_argument("net", help="the exported .bin net")
    p.add_argument("positions", help="datagen text file to take positions from")
    p.add_argument("--count", type=int, default=2000, help="positions to compare")
    p.add_argument("--max-mean-diff", type=float, default=5.0, help="fail above this mean |difference| in cp")
    p.add_argument("--evalfen", default=str(ROOT / "target" / "release" / "evalfen"))
    p.add_argument("--json", help="write the result here")
    args = p.parse_args()

    fens = []
    with open(args.positions) as f:
        for line in f:
            if line.strip():
                fens.append(line.split("|")[0].strip())
            if len(fens) == args.count:
                break

    python_cp = train.evaluate_fens(train.load_net(args.checkpoint), fens)
    engine = subprocess.run(
        [args.evalfen, f"nnue:{args.net}"], input="\n".join(fens) + "\n", capture_output=True, text=True, check=True
    )
    engine_cp = np.array([int(v) for v in engine.stdout.split()], dtype=np.float64)

    diff = np.abs(engine_cp - python_cp)
    result = {
        "positions": len(fens),
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "ok": bool(diff.mean() <= args.max_mean_diff),
    }
    worst = int(diff.argmax())
    print(f"{len(fens)} positions: mean |engine - pytorch| {result['mean_diff']:.2f} cp, max {result['max_diff']:.1f} cp")
    print(f"  worst: {fens[worst]}  engine {engine_cp[worst]:.0f}  pytorch {python_cp[worst]:.1f}")
    print("OK" if result["ok"] else f"MISMATCH: mean difference above {args.max_mean_diff} cp")
    if args.json:
        Path(args.json).write_text(json.dumps(result) + "\n")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
