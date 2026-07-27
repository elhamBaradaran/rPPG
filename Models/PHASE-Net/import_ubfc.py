"""
Helper - collect downloaded UBFC subjects into the folder the evaluator expects.

WHY: when you download folders from Google Drive, your browser saves them as ZIP
files (e.g. "subject1-20260723.zip" or "DATASET_2-001.zip"), usually in Downloads.
This script finds those, unpacks them, and arranges everything as:

    D:\\00-TU-CLAUSTHAL\\keiko-rppg\\UBFC\\subject1\\vid.avi
                                       \\subject1\\ground_truth.txt
                                       \\subject3\\...

It also handles already-extracted folders, and tells you which subjects are ready.

RUN:
    python import_ubfc.py                 # look in Downloads, unpack into the UBFC folder
    python import_ubfc.py --src "D:\\somewhere\\else"
    python import_ubfc.py --keep-zips     # do not delete the .zip after unpacking
    python import_ubfc.py --status        # just report what is ready, change nothing
"""

import argparse
import os
import re
import shutil
import sys
import zipfile

DEST = r"D:\00-TU-CLAUSTHAL\keiko-rppg\UBFC"
DEFAULT_SRC = os.path.join(os.path.expanduser("~"), "Downloads")


def status(dest=DEST):
    """Report which subjects are ready to evaluate."""
    if not os.path.isdir(dest):
        print(f"{dest} does not exist yet.")
        return []
    subs = sorted(
        [d for d in os.listdir(dest) if re.fullmatch(r"subject\d+", d)],
        key=lambda s: int(s.replace("subject", "")),
    )
    ready, partial = [], []
    total_mb = 0.0
    for s in subs:
        p = os.path.join(dest, s)
        v = os.path.join(p, "vid.avi")
        g = os.path.join(p, "ground_truth.txt")
        if os.path.isfile(v) and os.path.isfile(g):
            mb = os.path.getsize(v) / 1024 ** 2
            total_mb += mb
            ready.append((s, mb))
        else:
            have = []
            if os.path.isfile(v):
                have.append("vid.avi")
            if os.path.isfile(g):
                have.append("ground_truth.txt")
            partial.append((s, have))

    print(f"\nUBFC folder: {dest}")
    print(f"READY ({len(ready)} subject(s), {total_mb/1024:.1f} GB):")
    for s, mb in ready:
        print(f"   {s:<12} {mb:7.0f} MB")
    if partial:
        print(f"INCOMPLETE ({len(partial)}):")
        for s, have in partial:
            print(f"   {s:<12} has {have or 'nothing'}")
    if not ready:
        print("   (none yet)")
    return [s for s, _ in ready]


def _find_subject_dirs(root):
    """Yield every directory that directly contains vid.avi + ground_truth.txt."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if "vid.avi" in filenames and "ground_truth.txt" in filenames:
            yield dirpath


def _place(subject_dir, dest):
    """Move a found subject directory to dest/subjectN."""
    name = os.path.basename(subject_dir.rstrip("\\/"))
    m = re.search(r"subject\d+", name, re.IGNORECASE)
    if not m:
        # fall back to the parent folder name
        m = re.search(r"subject\d+", subject_dir, re.IGNORECASE)
    if not m:
        print(f"   ! cannot tell which subject this is: {subject_dir}")
        return None
    target = os.path.join(dest, m.group(0).lower())
    os.makedirs(target, exist_ok=True)
    moved = []
    for f in ("vid.avi", "ground_truth.txt"):
        src = os.path.join(subject_dir, f)
        dst = os.path.join(target, f)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue                      # already have it
        shutil.move(src, dst)
        moved.append(f)
    if moved:
        print(f"   + {os.path.basename(target)}: {', '.join(moved)}")
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="where your downloads are")
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--status", action="store_true", help="only report, change nothing")
    args = ap.parse_args()

    if args.status:
        status(args.dest)
        return

    if not os.path.isdir(args.src):
        sys.exit(f"Source folder not found: {args.src}")
    os.makedirs(args.dest, exist_ok=True)
    print(f"Looking for UBFC data in: {args.src}")

    # 1) unpack any zip that looks like it holds subject data
    tmp = os.path.join(args.dest, "_unzip_tmp")
    for entry in sorted(os.listdir(args.src)):
        if not entry.lower().endswith(".zip"):
            continue
        if not re.search(r"subject|ubfc|dataset", entry, re.IGNORECASE):
            continue
        zpath = os.path.join(args.src, entry)
        print(f"\n-> unpacking {entry} ({os.path.getsize(zpath)/1024**3:.2f} GB)")
        try:
            os.makedirs(tmp, exist_ok=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
        except zipfile.BadZipFile:
            print("   ! not a valid zip (still downloading?) - skipped")
            continue
        found = list(_find_subject_dirs(tmp))
        if not found:
            print("   ! no subject folders inside")
        for d in found:
            _place(d, args.dest)
        shutil.rmtree(tmp, ignore_errors=True)
        if not args.keep_zips:
            os.remove(zpath)
            print(f"   (deleted {entry} to free space)")

    # 2) also pick up already-extracted folders sitting in the source
    for d in _find_subject_dirs(args.src):
        print(f"\n-> found extracted folder: {d}")
        _place(d, args.dest)

    ready = status(args.dest)
    if ready:
        print("\nNext step - run the evaluation:")
        print(f'   python 04_ubfc_eval.py --data "{args.dest}"')
        print("   (add --delete-after to remove each video once it has been scored)")


if __name__ == "__main__":
    main()
