#!/usr/bin/env python3
"""
Batch Processing Pipeline for PCB OCR Dataset (Parallel)
=========================================================
3-phase pipeline maximising CPU/GPU utilisation:

  Phase 1 (parallel, CPU): Extract + Crop all boards
  Phase 2 (GPU, single):   Rotate ALL crops → RF-DETR (load model once)
  Phase 3 (parallel, CPU): Merge + Organise all boards

Usage:
    python process_all_zips.py [--workers 16] [--skip-existing] [--only-board X]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Force unbuffered output
print = partial(print, flush=True)

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_ROOT = PROJECT_ROOT / "Data"
ZIP_DIRS = [
    DATA_ROOT / "ocr_zip_files" / "20260223",
    DATA_ROOT / "ocr_zip_files" / "ocr_0305_all_boards",
]
OUTPUT_DIR = DATA_ROOT / "ocr_zip_files" / "0311_ocr"
EMPTY_DIR = DATA_ROOT / "ocr_zip_files" / "0311_ocr_empty"
EXISTING_OCR_DIR = DATA_ROOT / "0123_ocr"
CHECKPOINTS_ROOT = PROJECT_ROOT / "Checkpoints"
ORIENT_MODEL = CHECKPOINTS_ROOT / "orientation_classification" / "model.pth"
ORIENT_CONFIG = CHECKPOINTS_ROOT / "orientation_classification" / "config.yaml"
WORK_DIR = DATA_ROOT / "ocr_zip_files" / "_work"


# ============================================================================
# Helpers
# ============================================================================

def run_cmd(cmd, desc, timeout=600):
    """Run a subprocess; return True on success."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"  ERROR [{desc}]: {(r.stdout or '')[-300:]} {(r.stderr or '')[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{desc}]")
        return False
    except Exception as e:
        print(f"  EXCEPTION [{desc}]: {e}")
        return False


def extract_model_name_from_zip(zip_path: Path) -> Optional[str]:
    import zipfile, csv, io
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            pf = [n for n in zf.namelist() if n.endswith('products.csv')]
            if not pf:
                return None
            with zf.open(pf[0]) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, 'utf-8')):
                    mn = row.get('model_name', '').strip()
                    if mn:
                        return mn
    except Exception:
        pass
    return None


def collect_zip_files():
    zips = []
    for zd in ZIP_DIRS:
        if not zd.exists():
            continue
        for zf in sorted(zd.glob("*.zip")):
            zips.append((zf, zd.name))
        for sub in sorted(zd.iterdir()):
            if sub.is_dir():
                for zf in sorted(sub.glob("*.zip")):
                    zips.append((zf, zd.name))
    return zips


def dedup_boards(zip_files):
    model_to_zip: Dict[str, Tuple[Path, str]] = {}
    dupes = 0
    for zp, src in zip_files:
        mn = extract_model_name_from_zip(zp)
        if not mn or not mn.strip():
            continue
        if mn in model_to_zip:
            if src == "ocr_0305_all_boards":
                model_to_zip[mn] = (zp, src)
            dupes += 1
        else:
            model_to_zip[mn] = (zp, src)
    return model_to_zip, dupes


def _has_char_detections(json_path: Path) -> bool:
    try:
        with open(json_path) as f:
            data = json.load(f)
        return any(
            len(s.get("label", "")) == 1 and s.get("label") != " "
            for s in data.get("shapes", [])
        )
    except Exception:
        return False


def organize_board_output(crops_dir, merged_dir, board_name, output_dir, empty_dir):
    """Copy results into review-app-compatible structure. Returns (out, empty)."""
    board_crop_dir = crops_dir / board_name
    if not board_crop_dir.exists():
        return 0, 0

    fused_dir = output_dir / f"{board_name}_fused"
    candidate_dir = fused_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    board_empty_dir = empty_dir / board_name
    board_empty_dir.mkdir(parents=True, exist_ok=True)

    out_n = empty_n = 0
    for png in sorted(board_crop_dir.glob("*.png")):
        stem = png.stem
        jf = merged_dir / f"{stem}.json"
        if jf.exists() and _has_char_detections(jf):
            shutil.copy2(png, candidate_dir / f"{stem}.png")
            shutil.copy2(jf, candidate_dir / f"{stem}.json")
            out_n += 1
        else:
            shutil.copy2(png, board_empty_dir / f"{stem}.png")
            if jf.exists():
                shutil.copy2(jf, board_empty_dir / f"{stem}.json")
            empty_n += 1

    manifest_src = board_crop_dir / "crop_manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, fused_dir / "crop_manifest.json")

    if out_n == 0:
        shutil.rmtree(fused_dir, ignore_errors=True)
    if empty_n == 0:
        shutil.rmtree(board_empty_dir, ignore_errors=True)
    return out_n, empty_n


# ============================================================================
# Phase 1: Extract + Crop  (CPU only, parallel)
# ============================================================================

def phase1_single(model_name, zip_path, python_cmd):
    """Extract annotations from zip and crop _mount regions.

    Returns dict with status and paths.
    """
    board_work = WORK_DIR / model_name
    extract_dir = board_work / "extracted"
    crops_dir = board_work / "crops"

    try:
        if board_work.exists():
            shutil.rmtree(board_work)
        board_work.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Stage zip
        staging = board_work / "zip_staging"
        staging.mkdir()
        os.symlink(zip_path, staging / zip_path.name)

        # Extract
        ok = run_cmd([
            python_cmd, str(PROJECT_ROOT / "annotation_extraction.py"),
            "--data-dir", str(staging), "--output", str(extract_dir), "--skip-marker",
        ], f"extract({model_name})", timeout=120)
        if not ok or not list(extract_dir.glob("*.json")):
            return {"model_name": model_name, "status": "failed", "step": "extract"}

        # Crop
        ok = run_cmd([
            python_cmd, str(PROJECT_ROOT / "crop_components.py"),
            "--input", str(extract_dir), "--output", str(crops_dir),
        ], f"crop({model_name})", timeout=300)

        crop_dirs = [d for d in crops_dir.iterdir() if d.is_dir()] if crops_dir.exists() else []
        crop_count = sum(len(list(d.glob("*.png"))) for d in crop_dirs)

        if not ok or crop_count == 0:
            return {"model_name": model_name, "status": "failed", "step": "crop",
                    "reason": "no crops" if not crop_count else "crop error"}

        # Clean up extracted board image + JSONs to save disk (keep crops only)
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)

        return {
            "model_name": model_name, "status": "ok",
            "crops_dir": str(crops_dir),
            "crop_dirs": [str(d) for d in crop_dirs],
            "crop_count": crop_count,
        }
    except Exception as e:
        return {"model_name": model_name, "status": "failed", "step": "phase1", "reason": str(e)}


# ============================================================================
# Phase 2: Rotate + RF-DETR  (GPU, sequential — model loaded once)
# ============================================================================

def phase2_gpu(crop_dir_list, python_cmd):
    """Run rotation and dual-model RF-DETR on all crop directories.

    crop_dir_list: list of (model_name, [crop_board_dir, ...]) tuples.
    """
    # Collect ALL crop directories for a single rotation + detection pass
    all_crop_dirs = []
    for model_name, dirs in crop_dir_list:
        all_crop_dirs.extend(dirs)

    if not all_crop_dirs:
        return

    # --- Rotate: single call on WORK_DIR (recursive glob finds all PNGs) ---
    # rotate_img.py loads model once and processes all images recursively
    print(f"\n[Phase 2] Rotating all crops under {WORK_DIR} ...")
    t0 = time.time()
    run_cmd([
        python_cmd, str(PROJECT_ROOT / "rotate_img.py"),
        "--data-path", str(WORK_DIR),
        "--model-path", str(ORIENT_MODEL),
        "--config-path", str(ORIENT_CONFIG),
        "--device", "cuda",
    ], "rotate(all)", timeout=7200)
    print(f"  Rotation done in {time.time()-t0:.0f}s")

    # --- RF-DETR: one call per crop directory (model loaded per call) ---
    # To load the model only once, batch all images into a single input dir
    # by creating a combined dir with symlinks
    print(f"\n[Phase 2] Running dual RF-DETR on {len(all_crop_dirs)} directories...")
    t0 = time.time()

    # Create a combined input dir with symlinks to all crop images
    combined_dir = WORK_DIR / "_all_crops"
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Track original paths for later: {symlink_name -> (model_name, board_name, original_path)}
    image_map = {}
    for model_name, dirs in crop_dir_list:
        for crop_dir in dirs:
            board_name = Path(crop_dir).name
            for png in Path(crop_dir).glob("*.png"):
                # Prefix to avoid collision: {model_name}__{board_name}__{stem}.png
                link_name = f"{model_name}__{board_name}__{png.name}"
                link_path = combined_dir / link_name
                try:
                    os.symlink(png.resolve(), link_path)
                    image_map[link_name] = (model_name, board_name, str(png))
                except OSError:
                    shutil.copy2(str(png), str(link_path))

    total_images = len(image_map)
    print(f"  Combined {total_images} images into single input directory")

    # Run RF-DETR once for both models on the combined dir
    ok = run_cmd([
        python_cmd, str(PROJECT_ROOT / "run_rfdetr.py"),
        "-i", str(combined_dir),
        "--models", "char_224", "char_448",
        "--threshold", "0.5",
    ], "rfdetr(all)", timeout=3600)

    if not ok:
        print("  WARNING: RF-DETR failed on combined directory, falling back to per-board")
        # Fallback: per board directory
        for model_name, dirs in crop_dir_list:
            for cd in dirs:
                run_cmd([
                    python_cmd, str(PROJECT_ROOT / "run_rfdetr.py"),
                    "-i", str(cd),
                    "--models", "char_224", "char_448",
                    "--threshold", "0.5",
                ], f"rfdetr({Path(cd).name})", timeout=600)
        print(f"  RF-DETR fallback done in {time.time()-t0:.0f}s")
        return

    # Scatter detection results back to per-board directories
    det_224_dir = WORK_DIR / "_all_crops_char_224"
    det_448_dir = WORK_DIR / "_all_crops_char_448"

    for link_name, (model_name, board_name, orig_path) in image_map.items():
        stem = Path(link_name).stem  # model_name__board_name__orig_stem
        orig_stem = Path(orig_path).stem

        # Target directories for this board
        board_crops = WORK_DIR / model_name / "crops"
        target_224 = board_crops / f"{board_name}_char_224"
        target_448 = board_crops / f"{board_name}_char_448"
        target_224.mkdir(parents=True, exist_ok=True)
        target_448.mkdir(parents=True, exist_ok=True)

        # Move detection JSONs back, renaming to original stem
        for src_dir, tgt_dir in [(det_224_dir, target_224), (det_448_dir, target_448)]:
            if src_dir and src_dir.exists():
                src_json = src_dir / f"{stem}.json"
                if src_json.exists():
                    shutil.move(str(src_json), str(tgt_dir / f"{orig_stem}.json"))

    # Cleanup combined dirs
    shutil.rmtree(combined_dir, ignore_errors=True)
    shutil.rmtree(det_224_dir, ignore_errors=True)
    shutil.rmtree(det_448_dir, ignore_errors=True)

    print(f"  RF-DETR done in {time.time()-t0:.0f}s ({total_images} images)")


# ============================================================================
# Phase 3: Merge + Organise  (CPU, parallel)
# ============================================================================

def phase3_single(model_name, python_cmd):
    """Merge detections + organise output for one board. Returns result dict."""
    board_work = WORK_DIR / model_name
    crops_dir = board_work / "crops"
    merged_dir = board_work / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    result = {"model_name": model_name, "status": "success",
              "output_count": 0, "empty_count": 0}

    try:
        crop_dirs = [d for d in crops_dir.iterdir() if d.is_dir()
                     and not d.name.endswith("_char_224")
                     and not d.name.endswith("_char_448")]

        for cd in crop_dirs:
            board_name = cd.name
            det_224 = crops_dir / f"{board_name}_char_224"
            det_448 = crops_dir / f"{board_name}_char_448"
            manifest = cd / "crop_manifest.json"

            merge_args = [
                python_cmd, str(PROJECT_ROOT / "merge_detections.py"),
                "--output", str(merged_dir), "--images", str(cd),
            ]
            if det_224.exists():
                merge_args += ["--det-224", str(det_224)]
            if det_448.exists():
                merge_args += ["--det-448", str(det_448)]
            if manifest.exists():
                merge_args += ["--manifest", str(manifest)]

            ok = run_cmd(merge_args, f"merge({model_name}/{board_name})", timeout=300)
            if not ok:
                result["status"] = "failed"
                result["step"] = "merge"
                return result

            out_n, empty_n = organize_board_output(
                crops_dir, merged_dir, board_name, OUTPUT_DIR, EMPTY_DIR)
            result["output_count"] += out_n
            result["empty_count"] += empty_n

        # Cross-check with 0123_ocr
        if EXISTING_OCR_DIR.exists():
            existing = list(EXISTING_OCR_DIR.glob(f"{model_name}_*.json"))
            if existing:
                result["has_cross_check"] = True

        # Cleanup
        shutil.rmtree(board_work, ignore_errors=True)

    except Exception as e:
        result["status"] = "failed"
        result["reason"] = str(e)

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel PCB OCR pipeline.")
    parser.add_argument("--workers", "-w", type=int, default=16,
                        help="CPU workers for Phase 1 & 3 (default: 16)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-board", type=str, default=None)
    parser.add_argument("--max-boards", type=int, default=None)
    parser.add_argument("--python", type=str, default="python3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("PCB OCR Pipeline - Parallel Batch Processing")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    zip_files = collect_zip_files()
    print(f"Found {len(zip_files)} zip files")

    model_to_zip, dupes = dedup_boards(zip_files)
    print(f"Unique boards: {len(model_to_zip)} ({dupes} duplicates)")

    if args.only_board:
        if args.only_board in model_to_zip:
            model_to_zip = {args.only_board: model_to_zip[args.only_board]}
        else:
            print(f"Error: Board '{args.only_board}' not found")
            return

    boards = sorted(model_to_zip.items())
    if args.max_boards:
        boards = boards[:args.max_boards]

    if args.dry_run:
        print(f"\nDry run: would process {len(boards)} boards")
        for mn, (zp, src) in boards[:20]:
            print(f"  {mn} ({src})")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMPTY_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ================================================================
    # PHASE 1: Extract + Crop  (parallel CPU)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 1: Extract + Crop ({len(boards)} boards, {args.workers} workers)")
    print(f"{'='*80}")
    t0 = time.time()

    phase1_results = {}
    phase1_ok = []
    phase1_fail = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(phase1_single, mn, zp, args.python): mn
            for mn, (zp, src) in boards
        }
        for i, future in enumerate(as_completed(futures)):
            mn = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"model_name": mn, "status": "failed", "reason": str(e)}

            phase1_results[mn] = res
            if res["status"] == "ok":
                phase1_ok.append(mn)
            else:
                phase1_fail.append(mn)

            if (i + 1) % 20 == 0 or (i + 1) == len(boards):
                print(f"  Phase 1: {i+1}/{len(boards)} done "
                      f"({len(phase1_ok)} ok, {len(phase1_fail)} failed)")

    phase1_time = time.time() - t0
    total_crops = sum(phase1_results[mn].get("crop_count", 0) for mn in phase1_ok)
    print(f"\nPhase 1 done: {len(phase1_ok)} ok, {len(phase1_fail)} failed, "
          f"{total_crops} total crops in {phase1_time:.0f}s")

    if phase1_fail:
        print(f"  Failed: {phase1_fail[:10]}{'...' if len(phase1_fail)>10 else ''}")

    # ================================================================
    # PHASE 2: Rotate + RF-DETR  (GPU, single process)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 2: Rotate + RF-DETR ({len(phase1_ok)} boards, {total_crops} images)")
    print(f"{'='*80}")
    t0 = time.time()

    crop_dir_list = []
    for mn in phase1_ok:
        res = phase1_results[mn]
        crop_dir_list.append((mn, res["crop_dirs"]))

    phase2_gpu(crop_dir_list, args.python)
    phase2_time = time.time() - t0
    print(f"\nPhase 2 done in {phase2_time:.0f}s")

    # ================================================================
    # PHASE 3: Merge + Organise  (parallel CPU)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 3: Merge + Organise ({len(phase1_ok)} boards, {args.workers} workers)")
    print(f"{'='*80}")
    t0 = time.time()

    final_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(phase3_single, mn, args.python): mn
            for mn in phase1_ok
        }
        for i, future in enumerate(as_completed(futures)):
            mn = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"model_name": mn, "status": "failed", "reason": str(e)}
            final_results.append(res)

            if (i + 1) % 20 == 0 or (i + 1) == len(phase1_ok):
                print(f"  Phase 3: {i+1}/{len(phase1_ok)} done")

    phase3_time = time.time() - t0
    print(f"\nPhase 3 done in {phase3_time:.0f}s")

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - total_start
    success = sum(1 for r in final_results if r["status"] == "success")
    failed_p3 = sum(1 for r in final_results if r["status"] == "failed")
    total_out = sum(r.get("output_count", 0) for r in final_results)
    total_empty = sum(r.get("empty_count", 0) for r in final_results)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total boards:     {len(boards)}")
    print(f"Phase 1 ok/fail:  {len(phase1_ok)}/{len(phase1_fail)}")
    print(f"Phase 3 ok/fail:  {success}/{failed_p3}")
    print(f"Total crops:      {total_crops}")
    print(f"With chars:       {total_out}")
    print(f"Empty (no det):   {total_empty}")
    print(f"Phase 1 time:     {phase1_time:.0f}s")
    print(f"Phase 2 time:     {phase2_time:.0f}s")
    print(f"Phase 3 time:     {phase3_time:.0f}s")
    print(f"Total time:       {total_time/60:.1f} min")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Empty dir:        {EMPTY_DIR}")

    # Save log
    all_results = []
    for mn, (zp, src) in boards:
        entry = {"model_name": mn, "zip": zp.name, "source": src}
        if mn in phase1_results:
            entry["phase1"] = phase1_results[mn]
        p3 = [r for r in final_results if r.get("model_name") == mn]
        if p3:
            entry["phase3"] = p3[0]
        all_results.append(entry)

    log_path = OUTPUT_DIR / "processing_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Log:              {log_path}")

    # Failed boards
    all_failed = phase1_fail + [r["model_name"] for r in final_results if r["status"] == "failed"]
    if all_failed:
        print(f"\nFailed boards ({len(all_failed)}):")
        for mn in all_failed[:20]:
            reason = ""
            if mn in phase1_results and phase1_results[mn]["status"] == "failed":
                reason = phase1_results[mn].get("reason", phase1_results[mn].get("step", ""))
            else:
                p3 = [r for r in final_results if r.get("model_name") == mn and r["status"] == "failed"]
                if p3:
                    reason = p3[0].get("reason", p3[0].get("step", ""))
            print(f"  {mn}: {reason}")


if __name__ == "__main__":
    main()
