# -*- coding: utf-8 -*-
"""Download the four public training corpora (same sources as ZonUI-3B).

HF downloads go through hf-mirror by default (set HF_ENDPOINT otherwise).
Layout after running:  corpus/raw/<name>/...

  python data/download_corpus.py                 # everything except AMEX
  python data/download_corpus.py --only showui_web

AMEX is not on HF: follow data/download_corpus.py::AMEX_INSTRUCTIONS.
"""
import argparse
import os

HF_REPOS = {
    "showui_web": "showlab/ShowUI-web",          # 22K screenshots, per-element lists
    "showui_desktop": "showlab/ShowUI-desktop",  # 7.5K rows, per-element queries
    "uground": "osunlp/UGround-V1-Data-Box",  # Web-Hybrid bbox version; GATED: accept
                                              # terms on HF, then `huggingface-cli login`
}

AMEX_INSTRUCTIONS = """\
AMEX (mobile, 104K screenshots) is distributed by the authors, not on HF:
  1. request/download from the official page: https://www.yxchai.com/AMEX
  2. unpack into corpus/raw/amex/
  3. (optional, recommended) prepare metadata with ShowUI's helper:
     https://github.com/showlab/ShowUI/blob/main/prepare/hf_amex.py
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="corpus/raw")
    ap.add_argument("--only", choices=list(HF_REPOS) + ["amex"])
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download

    for name, repo in HF_REPOS.items():
        if args.only and args.only != name:
            continue
        dst = os.path.join(args.root, name)
        print("== %s -> %s" % (repo, dst))
        snapshot_download(repo, repo_type="dataset", local_dir=dst,
                          ignore_patterns=[".DS_Store", "**/.DS_Store", ".gitattributes"])
    if args.only in (None, "amex"):
        print(AMEX_INSTRUCTIONS)
    print("NOTE: extract ShowUI-web images with: tar -xzf corpus/raw/showui_web/images.tar.gz "
          "-C corpus/raw/showui_web/")


if __name__ == "__main__":
    main()
