"""Train the combat-evaluation network on sim-generated data.

Soft cross-entropy against the simulator's win/tie/loss distribution. Reports
val win%-MAE (how close the net's win probability is to the simulator's) and
argmax-outcome accuracy. Run:  python -m ml.train --train 8000 --epochs 40
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from .data import make_dataset
from .model import CombatValueNet


def _to_tensors(arrays):
    return [torch.from_numpy(a) for a in arrays]


def soft_ce(logits, target):
    """Cross-entropy with a distribution target (the sim's win/tie/loss)."""
    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def evaluate(model, xa, ma, xb, mb, y):
    model.eval()
    probs = model.predict_probs(xa, ma, xb, mb)
    win_mae = (probs[:, 0] - y[:, 0]).abs().mean().item()
    acc = (probs.argmax(-1) == y.argmax(-1)).float().mean().item()
    return win_mae, acc


def train(n_train=8000, n_val=1500, runs=80, epochs=40, lr=1e-3, batch=256,
          seed=0, verbose=True):
    if verbose:
        print(f"Generating data from the combat sim "
              f"({n_train}+{n_val} matchups, {runs} sims each)…")
    xa, ma, xb, mb, y = _to_tensors(make_dataset(n_train, runs, seed))
    vxa, vma, vxb, vmb, vy = _to_tensors(make_dataset(n_val, runs, seed + 999))

    model = CombatValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = xa.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(xa[idx], ma[idx], xb[idx], mb[idx])
            loss = soft_ce(logits, y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            mae, acc = evaluate(model, vxa, vma, vxb, vmb, vy)
            print(f"epoch {ep:3d}  train_loss {total / n:.4f}  "
                  f"val win%-MAE {mae:.3f}  val outcome-acc {acc:.3f}")
    mae, acc = evaluate(model, vxa, vma, vxb, vmb, vy)
    return model, {"win_mae": mae, "outcome_acc": acc}


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the combat value net")
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--val", type=int, default=1500)
    p.add_argument("--runs", type=int, default=80)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--save", help="path to save the trained weights (.pt)")
    a = p.parse_args(argv)
    model, metrics = train(a.train, a.val, a.runs, a.epochs)
    print("Final:", metrics)
    if a.save:
        torch.save(model.state_dict(), a.save)
        print("Saved ->", a.save)


if __name__ == "__main__":
    main()
