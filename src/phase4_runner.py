import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE4_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase4"
PHASE4_REPORT_DIR = PROJECT_ROOT / "reports" / "phase4"


def run_command(command):
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 multi-seed experimental evaluation runner."
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for this experiment.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    args = parser.parse_args()

    PHASE4_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PHASE4_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = args.seed

    print("\n" + "#" * 80)
    print(f"PHASE 4 EXPERIMENT — SEED {seed}")
    print("#" * 80)

    # ------------------------------------------------------------------
    # DQN baseline
    # ------------------------------------------------------------------

    dqn_artifact_dir = (
        PHASE4_ARTIFACT_DIR / f"dqn_seed_{seed}"
    )

    dqn_report_dir = (
        PHASE4_REPORT_DIR / f"dqn_seed_{seed}"
    )

    dqn_artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dqn_report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        [
            sys.executable,
            "src/baseline/train.py",
            "--episodes",
            str(args.episodes),
            "--steps-per-episode",
            str(args.steps_per_episode),
            "--replay-capacity",
            str(args.replay_capacity),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--artifact-dir",
            str(dqn_artifact_dir),
            "--report-dir",
            str(dqn_report_dir),
        ]
    )

    # ------------------------------------------------------------------
    # D3QN proposed method
    # ------------------------------------------------------------------

    d3qn_artifact_dir = (
        PHASE4_ARTIFACT_DIR / f"d3qn_seed_{seed}"
    )

    d3qn_report_dir = (
        PHASE4_REPORT_DIR / f"d3qn_seed_{seed}"
    )

    d3qn_artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    d3qn_report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        [
            sys.executable,
            "src/proposed/train.py",
            "--episodes",
            str(args.episodes),
            "--steps-per-episode",
            str(args.steps_per_episode),
            "--replay-capacity",
            str(args.replay_capacity),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--artifact-dir",
            str(d3qn_artifact_dir),
            "--report-dir",
            str(d3qn_report_dir),
        ]
    )

    print("\n" + "#" * 80)
    print(f"PHASE 4 TRAINING COMPLETE — SEED {seed}")
    print("#" * 80)

    print("\nDQN checkpoint:")
    print(dqn_artifact_dir / "baseline_dqn.pt")

    print("\nD3QN checkpoint:")
    print(d3qn_artifact_dir / "d3qn.pt")

    print("\nTraining reports:")
    print(dqn_report_dir / "baseline_training.json")
    print(d3qn_report_dir / "d3qn_training.json")


if __name__ == "__main__":
    main()