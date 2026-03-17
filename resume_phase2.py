#!/usr/bin/env python3
"""Resume batch pipeline from Phase 2 (Phase 1 already complete).

Reconstructs crop_dir_list from existing _work/ directory, then runs
optimized Phase 2 (single rotation call) + Phase 3 (parallel merge).
"""

import sys
import time
from functools import partial
from pathlib import Path

# Reuse everything from the main pipeline
from process_all_zips import (
    WORK_DIR, OUTPUT_DIR, EMPTY_DIR,
    phase2_gpu, phase3_single,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

print = partial(print, flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", "-w", type=int, default=16)
    parser.add_argument("--python", type=str, default="python3")
    args = parser.parse_args()

    print("=" * 80)
    print("PCB OCR Pipeline - RESUME from Phase 2")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    # Reconstruct crop_dir_list from _work/ directory
    # Structure: _work/{model_name}/crops/{board_name}/
    crop_dir_list = []
    model_names = []

    for model_dir in sorted(WORK_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        crops_parent = model_dir / "crops"
        if not crops_parent.exists():
            continue
        board_dirs = [
            str(d) for d in sorted(crops_parent.iterdir())
            if d.is_dir() and not d.name.endswith("_char_224")
            and not d.name.endswith("_char_448")
        ]
        if board_dirs:
            crop_dir_list.append((model_dir.name, board_dirs))
            model_names.append(model_dir.name)

    total_boards = len(crop_dir_list)
    total_crops = sum(
        sum(len(list(Path(d).glob("*.png"))) for d in dirs)
        for _, dirs in crop_dir_list
    )
    print(f"Found {total_boards} boards, {total_crops} crop images in {WORK_DIR}")

    total_start = time.time()

    # ================================================================
    # PHASE 2: Rotate + RF-DETR  (GPU, single process)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 2: Rotate + RF-DETR ({total_boards} boards, {total_crops} images)")
    print(f"{'='*80}")
    t0 = time.time()
    phase2_gpu(crop_dir_list, args.python)
    phase2_time = time.time() - t0
    print(f"\nPhase 2 done in {phase2_time:.0f}s")

    # ================================================================
    # PHASE 3: Merge + Organise  (parallel CPU)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 3: Merge + Organise ({total_boards} boards, {args.workers} workers)")
    print(f"{'='*80}")
    t0 = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMPTY_DIR.mkdir(parents=True, exist_ok=True)

    final_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(phase3_single, mn, args.python): mn
            for mn in model_names
        }
        for i, future in enumerate(as_completed(futures)):
            mn = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"model_name": mn, "status": "failed", "reason": str(e)}
            final_results.append(res)

            if (i + 1) % 20 == 0 or (i + 1) == total_boards:
                print(f"  Phase 3: {i+1}/{total_boards} done")

    phase3_time = time.time() - t0
    print(f"\nPhase 3 done in {phase3_time:.0f}s")

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - total_start
    success = sum(1 for r in final_results if r["status"] == "success")
    failed = sum(1 for r in final_results if r["status"] == "failed")
    total_out = sum(r.get("output_count", 0) for r in final_results)
    total_empty = sum(r.get("empty_count", 0) for r in final_results)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total boards:     {total_boards}")
    print(f"Phase 3 ok/fail:  {success}/{failed}")
    print(f"Total crops:      {total_crops}")
    print(f"With chars:       {total_out}")
    print(f"Empty (no det):   {total_empty}")
    print(f"Phase 2 time:     {phase2_time:.0f}s")
    print(f"Phase 3 time:     {phase3_time:.0f}s")
    print(f"Total time:       {total_time/60:.1f} min")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Empty dir:        {EMPTY_DIR}")

    if failed:
        print(f"\nFailed boards:")
        for r in final_results:
            if r["status"] == "failed":
                print(f"  {r['model_name']}: {r.get('reason', r.get('step', '?'))}")


if __name__ == "__main__":
    main()
