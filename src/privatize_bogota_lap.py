"""Generate synthetic Bogotá transaction tensors for demos (torch.randn)."""

import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--moving_window", type=int, default=0)
parser.add_argument("--saved_root", type=str, default="./Data/Processed/online")

args = parser.parse_args()

final_ret = torch.randn(80, 42 * 7)

os.makedirs(args.saved_root, exist_ok=True)
torch.save(
    final_ret,
    os.path.join(
        args.saved_root,
        "transaction_private_lap_{}_moving.pt".format(args.moving_window),
    ),
)
