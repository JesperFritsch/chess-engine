"""Check that the engine evaluates a net the way PyTorch does.

    python nnue/check.py nets/net.pt nets/net.bin data.txt

Takes positions from a datagen file and evaluates them three ways: with the
float network in PyTorch (from the checkpoint), with the exported net read back
and dequantized (the weights the engine actually holds, as floats), and with
the engine itself (via `evalfen` on the exported net).

Those three answer two separate questions, and mixing them up makes the check
useless:

* **engine vs dequantized** is the correctness test, and it is what this script
  passes or fails on. Both sides hold identical weights, so they can differ
  only by the engine's final integer division: half a centipawn, never more
  than one. Anything larger means the two implementations disagree about the
  network itself (a feature indexed differently, a view flipped the wrong way,
  a constant out of step), and the engine is playing a different net than the
  one that was trained.

* **float vs dequantized** is the quantization loss: what rounding the weights
  to integers costs. It is not a bug and it has no "correct" value. It grows
  with the size of the evaluations, so a net that has learned something shows
  more of it than a freshly initialized one -- single-digit centipawns on a
  trained net is normal. It is reported for information only.

Comparing float against the engine, as this script once did, measures the two
together. That hides a real mismatch on a weak net (where every evaluation is
near zero, so any two networks agree) and cries wolf on a strong one.
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
    p.add_argument("--max-mean-diff", type=float, default=1.5,
                   help="fail above this mean |engine - dequantized| in cp; the engine's integer "
                        "division alone accounts for about 0.5")
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

    float_cp = train.evaluate_fens(train.load_net(args.checkpoint), fens)
    deq_cp = train.evaluate_fens(train.load_exported(args.net), fens)
    engine = subprocess.run(
        [args.evalfen, f"nnue:{args.net}"], input="\n".join(fens) + "\n", capture_output=True, text=True, check=True
    )
    engine_cp = np.array([int(v) for v in engine.stdout.split()], dtype=np.float64)
    if len(engine_cp) != len(fens):
        sys.exit(f"evalfen returned {len(engine_cp)} evaluations for {len(fens)} positions")

    diff = np.abs(engine_cp - deq_cp)  # the correctness test
    quant = np.abs(float_cp - deq_cp)  # the price of quantization
    result = {
        "positions": len(fens),
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "quant_mean": float(quant.mean()),
        "quant_max": float(quant.max()),
        "eval_mean_abs": float(np.abs(float_cp).mean()),
        "ok": bool(diff.mean() <= args.max_mean_diff),
    }
    worst = int(diff.argmax())
    print(f"{len(fens)} positions, |eval| mean {result['eval_mean_abs']:.0f} cp")
    print(f"  engine vs dequantized  mean {result['mean_diff']:.2f} cp, max {result['max_diff']:.1f} cp"
          f"   <- implementations agree?")
    print(f"  float  vs dequantized  mean {result['quant_mean']:.2f} cp, max {result['quant_max']:.1f} cp"
          f"   <- quantization loss, for information")
    print(f"  worst: {fens[worst]}")
    print(f"    engine {engine_cp[worst]:.0f}  dequantized {deq_cp[worst]:.1f}  float {float_cp[worst]:.1f}")
    print("OK" if result["ok"] else
          f"MISMATCH: engine and PyTorch disagree about the network by {result['mean_diff']:.2f} cp "
          f"with identical weights (above {args.max_mean_diff})")
    if args.json:
        Path(args.json).write_text(json.dumps(result) + "\n")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
