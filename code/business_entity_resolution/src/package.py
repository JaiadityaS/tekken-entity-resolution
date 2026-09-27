"""Build <team>_submission.zip with the layout required by the organisers:

  output/{matching_results.tsv, candidate_pairs.tsv}
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md

Usage: python package.py <team_name>
"""
import sys
import zipfile
from pathlib import Path

from config import OUT_DIR, PROJECT

CODE = PROJECT / "code" / "business_entity_resolution"

if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "team"
    dst = PROJECT / f"{team}_submission.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(OUT_DIR / f, f"output/{f}")
        for f in sorted(CODE.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".prof" and f.name != "prof":
                z.write(f, f"code/business_entity_resolution/{f.relative_to(CODE).as_posix()}")
        z.write(PROJECT / "Documentation_template.md", "Documentation_template.md")
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
