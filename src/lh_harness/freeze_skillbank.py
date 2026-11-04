"""Create immutable SkillBank inputs for the existing benchmark evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path

from .skill_learning import SkillBank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path, help="Training SkillBank JSON")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    bank = SkillBank.load(args.bank)
    bank.export_frozen(args.output_dir / "validated_only.json")
    bank.export_frozen(args.output_dir / "validated_plus_provisional.json",
                       include_provisional=True)


if __name__ == "__main__":
    main()
