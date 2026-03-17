# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCB OCR dataset curation system. Builds clean OCR training datasets from component-crop images with printed strings (for retraining RF-DETR). A Flask web app provides human-in-the-loop review of auto-generated annotations in LabelMe JSON format.

**Pipeline:** ZIP extraction → annotation extraction → component cropping (subboard-aware) → orientation normalization → dual RF-DETR detection (224+448) → merge & NMS → auto-acceptance gate → subboard majority voting → human review via 3-level gallery UI.

**Key insight:** PCB boards contain identical subboards. Components at the same `region_group_id` across subboards are the same type — reviewing all siblings side-by-side in a grid is much faster than sequential review.

## Running the Application

```bash
# Main review app (4-tab UI: Gallery, Review, Dashboard, Queue)
# --dataset_root can be omitted and set via the web UI's dataset browser
python ocr_review_app.py --dataset_root Data/ocr_zip_files/0311_ocr --port 5001

# Full batch pipeline (parallel 3-phase)
python process_all_zips.py --zip-dir Data/ocr_zip_files/ --work-dir Data/_work/ --output Data/merged/

# Individual pipeline steps
python annotation_extraction.py --data-dir Data/raw/ --output Data/boards/
python crop_components.py --input Data/boards/ --output Data/crops/
python rotate_img.py --data-path Data/crops/ --model-path Checkpoints/orientation_classification/model.pth --config-path Checkpoints/orientation_classification/config.json
python run_rfdetr.py -i Data/crops/ --models char_224 char_448
python merge_detections.py --det-224 Data/crops_char_224/ --det-448 Data/crops_char_448/ --images Data/crops/ --manifest Data/crops/crop_manifest.json --output Data/merged/

# Resume pipeline from phase 2 (if phase 1 already complete)
python resume_phase2.py --work-dir Data/_work/ --output Data/merged/
```

## Testing

**Every code change must pass `node tests/test_e2e.js` before the user manually verifies.** See `tests/README.md` for full rules.

```bash
# Setup (one-time)
npm install puppeteer && npx puppeteer browsers install chrome

# Run (server must be running on port 5001)
node tests/test_e2e.js
```

41 Puppeteer tests covering all user operations + cross-component interaction logic. Tests never skip errors. If a test exposes a bug, fix the bug — don't skip the test.

## Architecture

### Pipeline Files

- **`process_all_zips.py`** — Batch orchestrator. 3-phase parallel pipeline: (1) extract+crop, (2) rotate+detect, (3) merge. Maximizes CPU/GPU utilization.
- **`resume_phase2.py`** — Resumes batch pipeline from phase 2 when phase 1 is already complete.
- **`annotation_extraction.py`** — Extracts board annotations from ZIP/CSV. Parses OBB shapes, component relationships (`region_group_id`, `array_index`).
- **`crop_components.py`** — Crops individual component regions from board images. Rotates OBB crops upright. Produces `crop_manifest.json` tracking subboard sibling relationships.
- **`rotate_img.py`** — Orientation normalization using classification model (0°/90°/180°/270°). Uses `third_party/daoai_classification/`.
- **`run_rfdetr.py`** — Batch character detection. Supports dual-model mode (`--models char_224 char_448`). Auto-detects model architecture/resolution from config. Skips background class.
- **`merge_detections.py`** — Merges dual-model outputs via NMS, assigns chars to lines, generates OCR text, applies acceptance gate, and performs subboard majority voting cross-validation.

### Core Layers (4 files)

- **`ocr_review_app.py`** — Flask app with REST API. Four-tab UI: Gallery (component group overview with batch accept/batch rotate), Review (always-interactive canvas with inline editing), Dashboard (KPIs + funnel), Queue (filtering + navigation). Multi-class support with board picker. Gallery comparison modal for subboard side-by-side review. Dataset browser for selecting datasets via web UI.
- **`line_loader.py`** — Parses LabelMe JSON into dataclasses (`ImageAnnotation` → `LineAnnotation` → `CharAnnotation` → `BBox`, `ComponentGroup`). Handles both old format (line_uid in descriptions) and RF-DETR format (plain confidence). Auto-assigns chars to lines by spatial containment.
- **`line_event_store.py`** — SQLite-backed event sourcing. Line lifecycle: PROPOSED → ACCEPTED/UNCERTAIN → REVIEWED/EDITED → DELETED. Group events: GROUP_ACCEPTED, GROUP_MAJORITY_ACCEPTED, INSTANCE_REJECTED. All mutations are immutable events with `region_group_id` tracking.
- **`metrics_calculator.py`** — Computes KPIs, acceptance funnel, failure reason breakdown, mismatch histogram, group-level metrics (leverage ratio, agreement stats).

### Supporting Files

- **`known_classlist.json`** — Reference list of known component class names.
- **`templates/ocr_review.html`** — Single-page dark-themed UI with four tabs. Gallery tab is primary. Review tab has unified always-interactive canvas (no separate edit mode overlay). All JS/CSS inline.
- **`third_party/daoai_classification/`** — Orientation classification model inference code.

## Key Conventions

- **Line UID format:** `{image_id}#L{index}` (e.g., `"img001#L0"`)
- **Image-level queue entry:** `__image__:{image_id}` for images with no lines (so users can add annotations)
- **Image ID:** JSON filename stem (no extension)
- **Character labels:** uppercase alphanumeric only `[0-9A-Z]`
- **LabelMe coordinates:** `[[x1,y1], [x2,y2]]` in shape points
- **Line status values:** `proposed`, `accepted`, `uncertain`, `reviewed`, `edited`, `deleted`
- **Review reasons:** `LEN_MISMATCH`, `NO_CHARS`, `NO_OCR`, `ORDER_AMBIGUOUS`, `BOX_ASSIGNMENT_SUSPECT`, `LOW_CONF`, `UNASSIGNED_CHARS`, `SIBLING_DISAGREE`
- **Character-to-line linking:** via `line_uid` stored in shape `description` field (semicolon-separated metadata), NOT via `group_id`
- **Subboard grouping:** via `region_group_id` from `crop_manifest.json` or shape descriptions (`group_{gid}_{idx}`). Falls back to text-based grouping when no manifest exists. Text grouping includes singletons.
- **Dataset structure:** `{dataset_root}/{BoardName}_fused/{candidate|final}/` contains paired `.png` + `.json` files
- **Checkpoint structure:** `Checkpoints/{model_name}/config.json` + `model.pth`
- **UI is localized to Chinese**

## Critical Implementation Rules

### Mutation Side-Effects Checklist

Every endpoint that modifies data (edit/add/delete/accept/rotate/finish) must handle these side-effects. Missing any one of these causes stale UI — the most common class of bugs in this codebase.

| Side-effect | When needed | Backend | Frontend |
|-------------|-------------|---------|----------|
| **Gallery cache** | Any data change | — | `invalidateGalleryCache()` in success callback |
| **Component groups** | Content changes (edit/add/delete text, NOT accept-only) | `load_manifest_and_groups()` | — |
| **Metrics** | Any status change | `invalidate_metrics()` (lazy — rebuilt on dashboard access) | — |
| **Thumbnail cache** | Image file changes (rotate, delete) | `thumbnail_cache.invalidate()` | — |
| **Queue** | Line add/delete, image delete | Rebuild queue with `IMAGE_QUEUE_PREFIX` for no-line images | `reloadQueueWithCurrentFilter()` |
| **Available subdirs** | `finish_review` creates `final/` | Add `'final'` to `app_state['available_subdirs']` | — |

### Null-Safety in JSON Data

LabelMe JSON fields that can be `null` and MUST be handled:
- **`group_id`** in shapes → use `s.get('group_id') or 0` (not `s.get('group_id', 0)` which returns None for null)
- **`imageWidth` / `imageHeight`** → skip bounds checks in `validate_bbox` when None
- **`bbox`** coordinates → check for None before comparison
- **`editLineBbox`** in frontend → guard all draw/resize code with `if (editLineBbox)`

### OCR Text vs Detection Text

- `ocr_text` = user-edited text (stored in line description as `ocr=...`)
- `det_text` = reconstructed from char labels (model output)
- **Always prefer `ocr_text` over `det_text`** in `line_loader.py` for majority voting: `line.ocr_text or line.det_text`
- On save, if `ocr_text` length matches char count, char labels auto-sync from `ocr_text` (backend `_accept_lines_in_instance` and edit endpoint)

### Browser Caching

- All image/thumb URLs include `_t=timestamp` for cache-busting (critical after rotations)
- Gallery API URLs include `_t=${Date.now()}`
- Thumbnail endpoint uses `Cache-Control: private, max-age=60` (short cache, URLs change on mutation)
- Full image endpoint uses `Cache-Control: no-cache` (may lack cache-buster in some paths)

### Flask Single-Thread Constraint

The dev server is single-threaded. Frontend `doSwitchClass()` must `await` the `switch_class` POST before firing `loadGallery()`, otherwise the GET arrives before the switch completes and returns stale data. All board/subdir switch functions use async/await for this reason.

### Canvas Drag at High Zoom

- `mousedown` must call `e.preventDefault()` to block scroll/text-select
- `mouseleave` must NOT cancel drag (user drags outside canvas at high zoom)
- Document-level `mouseup` handler ends drag even when mouse is outside canvas
- All resize mousedown paths must set `editDragStart` (not just move paths)

### Review Workflow (Two-Step)

1. **Accept** = mark lines as reviewed in SQLite event store + clear `needs_review` flag in JSON. Files stay in `candidate/`.
2. **Finish review** = move JSON + PNG from `candidate/` to `final/`. Only moves images where ALL lines are accepted/reviewed/edited.

"自动勾选一致组" only selects checkboxes — does NOT auto-accept. User must review and click "全部接受选中".

## Dependencies

Flask, OpenCV (`cv2`), NumPy, Pillow, sqlite3 (stdlib). For inference: `rfdetr`, `supervision`, `torch`, `torchvision`. For frontend testing: Puppeteer (`npm install puppeteer`).
