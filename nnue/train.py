"""Train the NNUE on records written by `pack`.

    python nnue/train.py data.bin [more.bin ...] --out nets/net.bin --epochs 10

Runs on CUDA when available, otherwise on the CPU (fine for small tests).

The network, for each position:

    white view: 768 inputs -> 256 accumulator \
                                                > [side to move | other side] -> clamp 0..1 -> 1 output
    black view: 768 inputs -> 256 accumulator /

Both views share the same first-layer weights: the black view is the board
flipped vertically with the colors swapped, so "my pawn on e2" is one feature
whichever color I am. The output is the evaluation for the side to move in
units of SCALE centipawns: `eval_cp = output * SCALE`.
"""

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURES = 768
NO_FEATURE = 768  # padding slot in the records; maps to a zero embedding row
HIDDEN = 256
SCALE = 400  # centipawns per output unit; also the sigmoid scale of the target

# Quantization for the engine: the accumulator in units of 1/QA, the output
# weights in units of 1/QB. Clipping weights to +-CLIP keeps every integer the
# engine computes inside i16 (accumulator) and i32 (output sum).
QA = 255
QB = 64
CLIP = 1.98

RECORD = np.dtype([("features", "<u2", 32), ("score", "<i2"), ("result", "u1"), ("stm", "u1")])
assert RECORD.itemsize == 68


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # An EmbeddingBag in "sum" mode is exactly a linear layer over a sparse
        # 0/1 input: it adds the weight rows of the features that are present.
        # Row 768 is the padding row and stays zero.
        self.ft = nn.EmbeddingBag(FEATURES + 1, HIDDEN, mode="sum", padding_idx=NO_FEATURE)
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.out = nn.Linear(2 * HIDDEN, 1)
        bound = 1 / math.sqrt(FEATURES)
        nn.init.uniform_(self.ft.weight, -bound, bound)
        with torch.no_grad():
            self.ft.weight[NO_FEATURE].zero_()

    def forward(self, us, them):
        acc_us = self.ft(us) + self.ft_bias
        acc_them = self.ft(them) + self.ft_bias
        hidden = torch.cat([acc_us, acc_them], dim=1).clamp(0, 1)  # CReLU
        return self.out(hidden).squeeze(1)

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-CLIP, CLIP)
            self.out.weight.clamp_(-CLIP, CLIP)


def flip(features):
    """White-view features -> black-view features: swap colors, mirror ranks."""
    pad = features == NO_FEATURE
    color, rest = features // 384, features % 384
    role, square = rest // 64, rest % 64
    flipped = (1 - color) * 384 + role * 64 + (square ^ 56)
    return np.where(pad, NO_FEATURE, flipped)


def batch(records, lam, device):
    """Inputs and targets for a batch, from the side to move's point of view."""
    white = records["features"].astype(np.int64)
    black = flip(white)
    stm_black = (records["stm"] == 1)[:, None]
    us = np.where(stm_black, black, white)
    them = np.where(stm_black, white, black)

    sign = np.where(records["stm"] == 1, -1.0, 1.0)
    score = records["score"].astype(np.float32) * sign
    result = records["result"].astype(np.float32) / 2  # 0, 0.5, 1 for White
    result = np.where(records["stm"] == 1, 1 - result, result)

    # Blend of what the search said and what happened in the game.
    target = lam / (1 + np.exp(-score / SCALE)) + (1 - lam) * result

    t = lambda a, dtype: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
    return t(us, torch.long), t(them, torch.long), t(target, torch.float32)


def export(net, path):
    """Quantized weights for the engine, little-endian:

    magic b"NNUE", u32 version 1, u32 hidden size, then
    ft weights  i16 [768][HIDDEN]   (x QA)
    ft bias     i16 [HIDDEN]        (x QA)
    out weights i16 [2 * HIDDEN]    (x QB; side to move's half first)
    out bias    i32                 (x QA * QB)
    """
    ft_w = net.ft.weight.detach().cpu().numpy()[:FEATURES]
    ft_b = net.ft_bias.detach().cpu().numpy()
    out_w = net.out.weight.detach().cpu().numpy()[0]
    out_b = net.out.bias.detach().cpu().numpy()[0]
    q = lambda a, s: np.round(a * s).astype("<i2")
    with open(path, "wb") as f:
        f.write(b"NNUE" + struct.pack("<II", 1, HIDDEN))
        f.write(q(ft_w, QA).tobytes())
        f.write(q(ft_b, QA).tobytes())
        f.write(q(out_w, QB).tobytes())
        f.write(struct.pack("<i", int(round(out_b * QA * QB))))


def load_exported(path):
    """The net as the *engine* holds it: the .bin read back, its integers turned
    into floats again.

    These are the exact weights the engine computes with, so a PyTorch forward
    pass over them differs from the engine's only by its final integer division
    (at most 1 cp). Comparing against this separates the two questions
    `check.py` has to answer: whether the two implementations agree about the
    network at all, and what the rounding to integers costs.
    """
    raw = Path(path).read_bytes()
    magic = raw[:4]
    version, hidden = struct.unpack("<II", raw[4:12])
    if magic != b"NNUE" or version != 1 or hidden != HIDDEN:
        raise ValueError(f"{path}: not a version-1 net with {HIDDEN} hidden "
                         f"(got {magic!r}, version {version}, hidden {hidden})")
    off = 12

    def take(count):
        nonlocal off
        a = np.frombuffer(raw, "<i2", count, off)
        off += count * 2
        return a.astype(np.float32)

    ft_w = take(FEATURES * HIDDEN).reshape(FEATURES, HIDDEN)
    ft_b = take(HIDDEN)
    out_w = take(2 * HIDDEN)
    (out_b,) = struct.unpack("<i", raw[off:off + 4])
    off += 4
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw)} bytes, expected {off}")

    net = Net()
    with torch.no_grad():
        net.ft.weight[:FEATURES] = torch.from_numpy(ft_w / QA)
        net.ft.weight[NO_FEATURE].zero_()
        net.ft_bias.copy_(torch.from_numpy(ft_b / QA))
        net.out.weight[0] = torch.from_numpy(out_w / QB)
        net.out.bias.fill_(out_b / (QA * QB))
    net.eval()
    return net


class Dataset:
    """Several record files used as one: the training part of each, plus a
    held-out tail of each for validation.

    Positions from one game sit next to each other in a file, so a random split
    would let the net "validate" on near-copies of training positions. Holding
    out the tail of every file keeps whole games apart (bar one per boundary).
    """

    def __init__(self, paths, val_fraction):
        self.train, self.val = [], []
        for path in paths:
            data = np.memmap(path, dtype=RECORD, mode="r")
            n_val = max(1, int(len(data) * val_fraction))
            self.train.append(data[:-n_val])
            self.val.append(data[-n_val:])
        self.offsets = np.cumsum([0] + [len(t) for t in self.train])

    def __len__(self):
        return int(self.offsets[-1])

    def gather(self, idx):
        """Training records at global indices `idx` (sorted, for locality)."""
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        parts = [self.train[f][idx[which == f] - self.offsets[f]] for f in np.unique(which)]
        return np.concatenate(parts)

    def validation(self, n):
        """Up to `n` validation records, spread over the files."""
        per_file = max(1, n // len(self.val))
        return np.concatenate([v[:per_file] for v in self.val])


def fen_records(fens):
    """Records (as `pack` writes them) for FENs; score and result are left 0.
    Lets the Python net be evaluated on exactly the positions the engine sees."""
    roles = "pnbrqk"
    records = np.zeros(len(fens), dtype=RECORD)
    for i, fen in enumerate(fens):
        board, stm = fen.split()[:2]
        features = []
        for r, rank in enumerate(board.split("/")):
            file = 0
            for ch in rank:
                if ch.isdigit():
                    file += int(ch)
                    continue
                square = (7 - r) * 8 + file
                color = 0 if ch.isupper() else 1
                features.append(color * 384 + roles.index(ch.lower()) * 64 + square)
                file += 1
        records[i]["features"] = features + [NO_FEATURE] * (32 - len(features))
        records[i]["stm"] = 0 if stm == "w" else 1
    return records


def load_net(checkpoint, device="cpu"):
    net = Net().to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device)["net"])
    net.eval()
    return net


def evaluate_fens(net, fens, device="cpu"):
    """Float evaluations in centipawns, side to move's point of view."""
    us, them, _ = batch(fen_records(fens), 1.0, device)
    with torch.no_grad():
        return (net(us, them) * SCALE).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="+", help="record files written by `pack`, used together")
    p.add_argument("--out", default="nets/net.bin", help="quantized net for the engine; checkpoint goes next to it as .pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--final-lr", type=float, default=1e-5, help="cosine decay to this by the last epoch")
    p.add_argument("--lambda", dest="lam", type=float, default=0.75, help="1.0 = search score only, 0.0 = result only")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--resume", help="continue an interrupted run from its checkpoint (.pt), optimizer and all")
    p.add_argument("--init", help="start from the weights of an earlier net's checkpoint (.pt), with a fresh optimizer")
    p.add_argument("--log", help="append one JSON line per epoch to this file")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data = Dataset(args.data, args.val_fraction)
    steps_per_epoch = max(1, len(data) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    n_val = sum(len(v) for v in data.val)
    print(f"{len(data)} training / {n_val} validation positions from {len(args.data)} file(s), "
          f"{steps_per_epoch} steps per epoch, device {device}", flush=True)

    net = Net().to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device)["net"])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.final_lr)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    val_batch = batch(data.validation(4 * args.batch_size), args.lam, device)

    for epoch in range(start_epoch, args.epochs):
        net.train()
        t0, total = time.time(), 0.0
        # A fresh shuffle every epoch; each position is seen once per epoch.
        order = rng.permutation(len(data))
        for step in range(steps_per_epoch):
            idx = np.sort(order[step * args.batch_size : (step + 1) * args.batch_size])
            us, them, target = batch(data.gather(idx), args.lam, device)
            pred = torch.sigmoid(net(us, them))
            loss = torch.mean((pred - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            net.clip()
            total += loss.item()

            # Progress within the epoch: an epoch over several generations of
            # data takes a while, and loop.py pipes this output, so without a
            # line now and then there is no way to tell training from a hang.
            if step % 20 == 0 or step + 1 == steps_per_epoch:
                rate = (step + 1) * args.batch_size / (time.time() - t0)
                print(f"\repoch {epoch + 1}/{args.epochs}  step {step + 1}/{steps_per_epoch}  "
                      f"train {total / (step + 1):.5f}  {rate:,.0f} pos/s   ", end="", flush=True)

        net.eval()
        with torch.no_grad():
            us, them, target = val_batch
            val_loss = torch.mean((torch.sigmoid(net(us, them)) - target) ** 2).item()
        secs = time.time() - t0
        stats = {
            "epoch": epoch + 1,
            "train_loss": total / steps_per_epoch,
            "val_loss": val_loss,
            "lr": sched.get_last_lr()[0],
            "pos_per_s": steps_per_epoch * args.batch_size / secs,
        }
        print(
            f"\repoch {epoch + 1}/{args.epochs}  train {stats['train_loss']:.5f}  val {val_loss:.5f}  "
            f"lr {stats['lr']:.2e}  {stats['pos_per_s']:,.0f} pos/s        ",
            flush=True,
        )
        if args.log:
            with open(args.log, "a") as f:
                f.write(json.dumps(stats) + "\n")
        torch.save(
            {"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1},
            out.with_suffix(".pt"),
        )
        export(net, out)

    print(f"wrote {out} (and checkpoint {out.with_suffix('.pt')})", flush=True)


if __name__ == "__main__":
    main()
