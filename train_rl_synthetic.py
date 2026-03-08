#!/usr/bin/env python3
"""Standalone RL traffic engineering training script using synthetic data.

Usage:
    python train_rl_synthetic.py [--timesteps 10000] [--output models/rl_traffic_engineer]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.rl_traffic_engineering import RLTrafficEngineer, _RL_AVAILABLE
from src.simulator.topology import NetworkTopology
from src.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL traffic engineering agent on synthetic data")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Total PPO training timesteps (default: 10000; use 50000 for a stronger policy)",
    )
    parser.add_argument(
        "--output",
        default="models/rl_traffic_engineer",
        help="Output path for the saved model (SB3 appends .zip automatically)",
    )
    args = parser.parse_args()

    if not _RL_AVAILABLE:
        print("ERROR: stable-baselines3 and gymnasium are required.")
        print("Install with: pip install stable-baselines3[extra] gymnasium")
        sys.exit(1)

    print("Building ISP network topology...")
    topology = NetworkTopology()
    link_ids = [f"{u}-{v}" for u, v in topology.get_graph().edges()]
    print(f"  {len(link_ids)} links in topology: {link_ids[:5]}{'...' if len(link_ids) > 5 else ''}")

    print(f"\nInitialising RLTrafficEngineer (timesteps={args.timesteps})...")
    rl_engineer = RLTrafficEngineer(link_ids=link_ids, total_timesteps=args.timesteps)

    print("Training PPO agent on synthetic NetworkSimEnv...")
    rl_engineer.train()
    print("Training complete.")

    print(f"\nSaving model to {args.output} ...")
    rl_engineer.save(args.output)
    print(f"Done. Model saved to: {os.path.abspath(args.output)}.zip")
    print("\nTo use the model, run the main system — DeciderAgent auto-loads it on startup.")


if __name__ == "__main__":
    main()
