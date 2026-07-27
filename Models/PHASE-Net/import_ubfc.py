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
    """Yield every directory holding EITHER vid.avi or ground_truth.txt.

    Google Drive splits large downloads by size, so a subject's video and its
    ground truth often land in DIFFERENT zips. We therefore collect whatever
    pieces we find and let them merge into the same destination folder.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        if "vid.avi" in filenames or "ground_truth.txt" in filenames:
            yield dirpath


def _zip_holds_ubfc(zpath):
    """Look INSIDE a zip instead of guessing from its filename.

    Google Drive names bulk downloads 'drive-download-<timestamp>-1-005.zip',
    which says nothing about the contents - so we check the file list instead.
    Returns True / False, or None if the zip cannot be read (still downloading).
    """
    try:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
    except (zipfile.BadZipFile, OSError):
        return None
    return any(n.endswith("vid.avi") or n.endswith("ground_truth.txt")
               for n in names)


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
        if not os.path.isfile(src):
            continue                      # this zip only had the other piece
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue                      # already have it from an earlier zip
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

    # 1) unpack every zip whose CONTENTS look like UBFC data
    tmp = os.path.join(args.dest, "_unzip_tmp")
    zips = [e for e in sorted(os.listdir(args.src)) if e.lower().endswith(".zip")]
    print(f"found {len(zips)} zip file(s) to inspect")

    for entry in zips:
        zpath = os.path.join(args.src, entry)
        gb = os.path.getsize(zpath) / 1024 ** 3
        holds = _zip_holds_ubfc(zpath)
        if holds is None:
            print(f"\n-  {entry}: unreadable (still downloading?) - skipped")
            continue
        if not holds:
            print(f"\n-  {entry}: no UBFC data inside - skipped")
            continue

        free_gb = shutil.disk_usage(args.dest).free / 1024 ** 3
        if free_gb < gb * 1.5:
            print(f"\n!  {entry}: only {free_gb:.1f} GB free, need ~{gb*1.5:.1f} GB - stopping")
            break

        print(f"\n-> unpacking {entry} ({gb:.2f} GB, {free_gb:.1f} GB free)...")
        try:
            os.makedirs(tmp, exist_ok=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
        except (zipfile.BadZipFile, OSError) as e:
            print(f"   ! extract failed: {e}")
            shutil.rmtree(tmp, ignore_errors=True)
            continue

        found = list(_find_subject_dirs(tmp))
        placed = 0
        for d in found:
            if _place(d, args.dest):
                placed += 1
        shutil.rmtree(tmp, ignore_errors=True)

        if not found:
            print("   ! no subject folders inside after all - zip kept")
            continue
        # Only remove the download once its contents are safely in place.
        if not args.keep_zips and placed:
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
