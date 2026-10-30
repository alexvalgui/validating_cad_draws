"""
Find (and optionally delete) STEP files that are truncated/corrupt --
e.g. left behind by a killed/timed-out/OOM'd conversion process before
stl_to_step.py wrote atomically.

A valid STEP file must end with "END-ISO-10303-21;" (the standard
terminator). Anything missing it is incomplete, regardless of why.

Usage:
    python find_corrupt_steps.py <dir> [--apply]
"""
import sys
from pathlib import Path


def is_corrupt(path: Path) -> bool:
    try:
        # STEP files can be large; only need to check the tail.
        size = path.stat().st_size
        if size == 0:
            return True
        with open(path, "rb") as f:
            f.seek(max(0, size - 200))
            tail = f.read().decode("utf-8", errors="ignore")
        return "END-ISO-10303-21" not in tail
    except Exception:
        return True


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target_dir = Path(sys.argv[1])
    apply = "--apply" in sys.argv[2:]

    all_steps = sorted(target_dir.rglob("*.step")) + sorted(target_dir.rglob("*.stp"))
    corrupt = [p for p in all_steps if is_corrupt(p)]

    print(f"Checked {len(all_steps)} STEP files, found {len(corrupt)} corrupt/truncated:")
    for p in corrupt:
        print(f"  {p}  ({p.stat().st_size} bytes)")
        if apply:
            p.unlink()

    if not apply and corrupt:
        print("\nDry run -- rerun with --apply to delete these so "
              "stl_to_step.py will regenerate them.")


if __name__ == "__main__":
    main()