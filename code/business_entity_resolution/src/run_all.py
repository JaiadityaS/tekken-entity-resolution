"""Run the full pipeline end-to-end (data -> blocking -> matching -> output)."""
import subprocess
import sys

STEPS = [
    ["prepare.py", "all"],
    ["make_candidates.py", "train"],
    ["make_candidates.py", "test"],
    ["train.py"],
    ["predict.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(">>", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], check=True)
