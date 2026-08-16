"""Single entrypoint: python run_all.py"""

import os
import sys
import subprocess

from src.utils import set_seed

SEED = 42  # fixed seed for reproducibility (no course STUDENT_ID needed)


def main() -> None:
    set_seed(SEED)

    print(f"== HW1 run_all (seed={SEED}) ==")

    print("-- running tests/ --")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
    if result.returncode != 0:
        print("Tests FAILED -- stopping before training.")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)

    print("-- experiments/train_best.py --")
    from experiments.train_best import main as train_best_main
    test_acc = train_best_main(SEED)

    print("-- experiments/compare_optimizers.py --")
    from experiments.compare_optimizers import main as compare_main
    compare_main(SEED)

    print(f"== FINAL TEST ACCURACY: {test_acc:.4f} ==")


if __name__ == "__main__":
    main()
