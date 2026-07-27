"""
Fetch missing ground_truth.txt files for UBFC subjects that already have a video.

WHY THIS EXISTS
  Downloading UBFC folders from Google Drive splits them by size, so a subject's
  vid.avi and its ground_truth.txt often end up in different archives - and some
  ground-truth files simply never arrive. A video without ground truth cannot be
  scored, so those subjects would be wasted.

  Ground-truth files are tiny (~74 KB), and Google's "too many downloads" quota
  only bites on the large video files, so these can be fetched directly.

USAGE
    python fetch_ground_truth.py            # fill in whatever is missing
    python fetch_ground_truth.py --dry-run  # just show what would be fetched
"""

import argparse
import os
import re
import sys

UBFC = r"D:\00-TU-CLAUSTHAL\keiko-rppg\UBFC"

# Google Drive file IDs of ground_truth.txt, read from the official
# UBFC_DATASET/DATASET_2 shared folder.
GT_IDS = {
    "subject1":  "1q7LX_8Ggfl43ZmsFXMrUkV2hNnKr-yTE",
    "subject3":  "1tKRzecjw14TGFkizo4XvMEWXeefriiCH",
    "subject4":  "1vvyDn_3hzw-AvgX69Ct3O2omH_D_erHj",
    "subject5":  "1yD3ng0b2D5gFdFMfmhOphCjunXeuU4IB",
    "subject8":  "1yKqVlkTkBp93KXUvmniOZ3JVfK-HxNnP",
    "subject9":  "1yW-OciEs79xm3Me5AFqbsCIipZrj4wwZ",
    "subject10": "1qLMw4RCX7_n87XaE_e-LpI8pFZ3h6ErX",
    "subject11": "1qSZgCAC77QgDWMrLtnjyNSOL-9h7dJJ4",
    "subject12": "1qeLIBTBRULL6k283pQMsdco4ZKzG8wDc",
    "subject13": "1qsUcZOvEd-A_e_Niay5idM62ron1Rbvg",
    "subject14": "1rFS0-3aOpD2rNvKclr3_G41NAR29mfKn",
    "subject15": "1rHZzYZlSt1yST-BDjbGYspwZ_tNlLlU_",
    "subject16": "1rgnJuHy3qik-LrYmpb4-dk84hiXSrzTf",
    "subject17": "1rojrR7jH64wFOlvvgYpxg5BHyQdEungD",
    "subject18": "1s2XxazUfBaI-QEXhGdwLRBoavAGxfQL0",
    "subject20": "1s5YThjp7ZkDjv9BXsnhf1n-xSFCmLzDP",
    "subject22": "1sThbS1zW-7gMw7x9_MAqoq3P7rJ4l-m5",
    "subject23": "1skOIpdO3cy8WceTQtdlGZVIqrol4_3hn",
    "subject24": "1swEE0XiJzAYwq9rk7TGBVgomZVhr4p4a",
    "subject25": "1t-8g3-2tvtgyNLK6v4ubMRRpOQTDdxTI",
    "subject26": "1t5v2Q6F38rSZ9gLfBXZ-Uo7UcUe1hvxf",
    "subject27": "1tCe_6Gshg-wTeT3UaNnfeC-fWAAZqZY3",
    "subject30": "1tXJX_zCrzE92GznFcPSuD5KyqnQFuGo3",
    "subject31": "1twhHHY91hjG2rPiV2IefWXca9VbDVRfS",
    "subject32": "1u3ziH9Fmz5CnwtBCG8BQL1Fgo4P8ii6J",
    "subject33": "1uMiq9i6ushno754vimuxa3zT1O5RUeLW",
    "subject34": "1uXIEHosBTbcfcjDWeQWSquK5Qw9XNGXU",
    "subject35": "1uhR8uFjLRg-zoaPK02-3CHUXl4sc7oMP",
    "subject36": "1v6cU8zGHOlNsdJDOE8657zTVwn35WiBB",
    "subject37": "1vIjv_lmxPpqz5cJC-gk_oCF2A7zd7EBa",
    "subject38": "1vRnzklzv1lhfPHNo8LztBF1RJ65Z719b",
    "subject39": "1vf8Jl0teyMCDExz4WAD5xrL-PM7UBPp0",
    "subject40": "1w390Xlz-dQKVDxx4otan9pbIwXVOrldR",
    "subject41": "1wHy0jozHq04aJY1orom4zX09ALVn6f6U",
    "subject42": "1w_1W7ilPBCmuJp_xSbUAiielLgB7zxHV",
    "subject43": "1wl-JPJ2I6kbfOvCEkEHaxqifRSYhWxSN",
    "subject44": "1x3yxN1b6zjokRW2VoAyAstzlC5izIXcA",
    "subject45": "1xEjWcpunrXDiIPmvLfS1VSEfWEQckFJV",
    "subject46": "1xItAgY8Air0onXVK_eM_FK4idoTdPLEJ",
    "subject47": "1xZGV8XsgMg_XX5cuFUQpLCXfNsHF-2UP",
    "subject48": "1xzTUqwcVYHRLyApEXEGVOfA0PAtqHDiG",
    "subject49": "1y7_UASK0V1-a8dXhsrtLnwhod90Wfzs2",
}


def looks_valid(path):
    """A real UBFC ground_truth.txt starts with a long line of float BVP values."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        vals = [v for v in first.split() if v]
        float(vals[0])
        return len(vals) > 100
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubfc", default=UBFC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.ubfc):
        sys.exit(f"UBFC folder not found: {args.ubfc}")

    subs = sorted(
        [d for d in os.listdir(args.ubfc) if re.fullmatch(r"subject\d+", d)],
        key=lambda s: int(s.replace("subject", "")),
    )

    need = []
    for s in subs:
        vid = os.path.join(args.ubfc, s, "vid.avi")
        gt = os.path.join(args.ubfc, s, "ground_truth.txt")
        if os.path.isfile(vid) and not looks_valid(gt):
            need.append(s)

    print(f"{len(subs)} subject folder(s); {len(need)} need ground truth: {need}")
    if not need:
        print("Nothing to do - every video already has its ground truth.")
        return
    if args.dry_run:
        print("(dry run - nothing downloaded)")
        return

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required:  pip install gdown")

    ok, failed = [], []
    for s in need:
        fid = GT_IDS.get(s)
        if not fid:
            print(f"  {s}: no known file id - skipped")
            failed.append(s)
            continue
        out = os.path.join(args.ubfc, s, "ground_truth.txt")
        try:
            gdown.download(id=fid, output=out, quiet=True)
        except Exception as e:
            print(f"  {s}: download failed ({type(e).__name__})")
            failed.append(s)
            continue
        if looks_valid(out):
            print(f"  {s}: OK ({os.path.getsize(out)/1024:.0f} KB)")
            ok.append(s)
        else:
            print(f"  {s}: downloaded file does not look like ground truth")
            if os.path.isfile(out):
                os.remove(out)
            failed.append(s)

    print(f"\nfetched {len(ok)}, failed {len(failed)}")
    if failed:
        print(f"failed: {failed}")
        print("Google may be rate-limiting; try again in a few minutes.")


if __name__ == "__main__":
    main()
