# OCR Dataset Curation Pipeline — Redesign Plan

## Context

Build clean OCR training datasets from PCB component crop images for retraining RF-DETR. The old pipeline relied on Hunyuan OCR (now removed). The new pipeline uses only:

- **Orientation classification model** (`Checkpoints/orientation_classification/`) — predicts 0°/90°/180°/270°
- **RF-DETR 224 model** (`Checkpoints/char_224/`) — detects characters [0-9, A-Z] + line-bboxes at 224px
- **RF-DETR 448 model** (`Checkpoints/char_448/`) — same classes at 448px

**Key insight:** PCB boards contain identical subboards (array_index). Components at the same `region_group_id` across subboards are the same component type at the same position — same text layout, but individual instances may have stains, scratches, or print defects that corrupt characters. Therefore, siblings must be **reviewed together side-by-side** (not blindly propagated). Grouping enables:
- **Majority voting**: if 15/16 instances detect the same OCR text, the 1 outlier is flagged
- **Visual comparison**: reviewer sees all instances at once, spots defects quickly
- **Cross-validation**: disagreement between siblings highlights detection errors
- Real data: 1–100 subboards per board (e.g., 16 subboards × 27 components = reviewer checks 27 comparison grids instead of 432 sequential images)

---

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────┐
│ Stage 0: Extract Annotations                            │
│   annotation_extraction.py (existing)                   │
│   ZIP → board image + component OBB bboxes (LabelMe)    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Crop Components (NEW: crop_components.py)      │
│   Board image + OBBs → individual component crops       │
│   Groups by region_group_id (subboard-aware)            │
│   Rotates OBB upright before cropping                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Orientation Normalization                       │
│   rotate_img.py (existing)                              │
│   Rotate crops to canonical left-to-right orientation   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Character + Line Detection                     │
│   run_rfdetr.py (fix config key)                        │
│   Run both 224 and 448 models → two sets of JSONs       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Merge & Auto-Accept (NEW: merge_detections.py) │
│   Merge 224+448 outputs (NMS)                           │
│   Assign chars → lines (spatial containment)            │
│   Auto-generate OCR text from char labels               │
│   Auto-accept high-confidence, flag uncertain           │
│   Group by region_group_id for subboard propagation     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Human Review (REDESIGN: ocr_review_app.py)     │
│   Gallery view → component groups quick overview         │
│   Subboard comparison: review all siblings side-by-side  │
│   Majority voting flags outliers automatically           │
│   Drill-down for corrections only                       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 6: Final Dataset                                  │
│   Paired .png + .json files (Data/0123_ocr/ format)     │
└─────────────────────────────────────────────────────────┘
```

---

## Stage 1: `crop_components.py` (NEW)

### Purpose
Crop individual component regions from full board images. Rotate OBB crops upright. Track subboard relationships.

### Input
- Board image (`model_name.jpg`) from `annotation_extraction.py` output
- LabelMe JSON (`model_name.json`) with `rotated_box` shapes

### Output
- `{output_dir}/{board_name}/{ComponentClass}_{group_id}_{array_index}.png` — crop image
- `{output_dir}/{board_name}/crop_manifest.json` — metadata mapping:
```json
{
  "board_name": "model_name",
  "crops": [
    {
      "crop_file": "Resistor_1769841526_0.png",
      "component_class": "Resistor",
      "region_group_id": "1769841526",
      "array_index": "0",
      "siblings": ["Resistor_1769841526_1", "Resistor_1769841526_2"]
    }
  ]
}
```

### Crop Logic
1. Load board image and JSON
2. For each shape with `_full` or `_mount` label:
   - Parse OBB: center, width, height, angle from `points` + `angle` field
   - Compute affine rotation matrix around OBB center
   - Warp image region, crop axis-aligned rect → upright component image
   - Parse `region_group_id` and `array_index` from shape `description` (format: `group_{gid}_{idx}`)
3. Build `crop_manifest.json` with sibling relationships (same `region_group_id`, different `array_index`)

### Reuse from `annotation_extraction.py`
- `parse_obb_shape()` — parse OBB geometry
- `_frame_dimensions()` — get width/height from OBB
- `extract_label_prefix()` — normalize component class
- `rotate_point()` — coordinate rotation math

---

## Stage 3: Fix `run_rfdetr.py`

### Changes
1. **Fix config key** (line 110): `config['DATASET']['CLS_LABELS']` → handle both formats:
   ```python
   labels_list = config.get('class_labels', config.get('DATASET', {}).get('CLS_LABELS', []))
   ```
2. **Skip background class**: The config has `"background"` at index 0. RF-DETR class_id 0 = background → skip it.
   ```python
   class_labels = {i: name for i, name in enumerate(labels_list) if name != 'background'}
   ```
3. **Batch mode for dual models**: Add `--models` argument to run both 224 and 448 in sequence:
   ```bash
   python run_rfdetr.py -i crops/ --models char_224 char_448 --output-prefix det
   # Outputs: det_224/*.json and det_448/*.json
   ```

---

## Stage 4: `merge_detections.py` (NEW)

### Purpose
Merge dual-model RF-DETR outputs. Assign characters to lines. Auto-generate OCR text. Apply acceptance gate. Handle subboard propagation.

### Input
- `{det_224_dir}/*.json` — 224 model detections
- `{det_448_dir}/*.json` — 448 model detections
- `{images_dir}/` — crop images (for dimensions)
- `crop_manifest.json` — subboard relationships

### Output
- `{output_dir}/{image_name}.json` — merged LabelMe JSON with enriched descriptions:
  - Line: `"line_uid={img}#L{i};ocr={text};conf={c};needs_review={0|1};reason={...}"`
  - Char: `"line_uid={img}#L{i};idx={j};conf={c}"`
- `{output_dir}/{image_name}.png` — copy/symlink of crop image

### Merge Algorithm

```
1. LOAD both detection JSONs for same image

2. MERGE CHARACTERS (NMS across models):
   - Pool all character detections from both models
   - Sort by confidence descending
   - For each char, suppress lower-confidence chars with IoU > 0.5
   - Keep surviving detections

3. MERGE LINE-BBOXES (NMS across models):
   - Same NMS approach as characters
   - If two line-bboxes overlap (IoU > 0.3), merge into union bbox

4. ASSIGN CHARACTERS TO LINES:
   - For each character, find line-bbox whose area contains char center
   - If char center inside multiple lines → assign to line with most overlap
   - Unassigned chars → create new line-bbox around them (or flag)

5. SORT & GENERATE OCR TEXT:
   - Within each line, sort chars by x-center (left to right)
   - Concatenate labels → ocr_text (e.g., "2001")
   - Compute line confidence = min(char confidences)

6. AUTO-ACCEPTANCE GATE:
   - Auto-accept if:
     - All chars in line have confidence > 0.8
     - No spatial ambiguity (chars well-separated, clear left-to-right order)
     - At least 1 character in line
   - Flag for review (needs_review=1) with reason if:
     - Any char confidence < 0.8 → "LOW_CONF"
     - Chars not clearly ordered → "ORDER_AMBIGUOUS"
     - Line has 0 chars → "NO_CHARS"
     - Unassigned chars exist → "UNASSIGNED_CHARS"

7. WRITE output JSON with line_uid and metadata in descriptions
```

### Subboard Cross-Validation
After merge, group outputs by `region_group_id` from `crop_manifest.json`:
1. **Majority voting**: For each group, compare auto-OCR text across all siblings. If majority agree → high confidence. If any disagree → flag the outliers with reason `"SIBLING_DISAGREE"`.
2. **Confidence aggregation**: Group confidence = median of per-instance line confidences. Low-confidence groups surface first in review.
3. **Outlier detection**: Instances where detection differs from majority (different char count, different labels) are pre-flagged for individual review.

---

## Stage 5: Redesign Human Review (`ocr_review_app.py`)

### New Workflow: 3-Level Review

**Level 1 — Component Group Gallery (fastest)**
- Show one representative thumbnail per `region_group_id`
- Display: crop image + auto-OCR text + confidence + subboard count badge ("×16") + agreement indicator
- Agreement indicator: "16/16 agree" (green) vs "14/16 agree, 2 outliers" (orange)
- Sort: disagreement first, then low confidence, then unreviewed
- Batch select + accept: approve groups where all siblings agree
- Color coding: green (all agree, high conf), orange (some disagree), red (low confidence)
- Keyboard: ←→ navigate, A accept group, Enter drill down

**Level 2 — Subboard Comparison Grid (core review)**
- Show ALL sibling instances side by side in a grid
- Each instance shows: crop image, auto-OCR text per line, char bboxes drawn, confidence
- Highlight outliers (instances that disagree with majority) with red border
- Reviewer visually scans grid — can spot stains/scratches/defects quickly across all instances
- Actions per instance: accept, reject (mark defective), edit (drill to Level 3)
- Action for entire group: "Accept All" (if all look good), "Accept Majority" (reject outliers)
- Show majority consensus text prominently at top of grid

**Level 3 — Line Edit Mode (slow, only when needed)**
- Existing bbox editing for individual corrections
- Pre-filled OCR text from auto-generation (or majority consensus text)
- Character label editing, bbox resize/move/add/delete
- Keep existing keyboard shortcuts

### Key Acceleration
| Scenario | Old workflow | New workflow |
|----------|-------------|-------------|
| 16 subboards × 27 components, all agree | Review 432 images one-by-one | Scan 27 comparison grids, batch accept |
| 100 subboards × 1 component, 2 defective | Review 100 images sequentially | 1 grid, spot 2 outliers, reject them |
| 6 subboards × 191 components, mixed | Review 1146 images | 191 grids, most quick-accept |
| High confidence, all siblings agree | Review 1 line at a time | Batch accept from gallery without opening |

### Backend API Changes

**New endpoints:**
- `GET /api/gallery?page=1&sort=confidence&filter=needs_review` — paginated component groups with thumbnails
- `GET /api/group/<region_group_id>` — all subboard instances for a component group
- `POST /api/group/accept` — accept entire group (all siblings verified)
- `POST /api/group/accept_majority` — accept majority, reject outliers
- `POST /api/instance/reject` — reject specific instance (defective)
- `POST /api/batch/accept` — accept multiple groups at once
- `GET /api/image/<image_id>/full` — full image with all lines and chars rendered

**Modified endpoints:**
- `/api/line/<line_uid>` — keep for line-level edit drill-down
- `/api/queue` — add grouping and confidence sorting

### Frontend Changes (`templates/ocr_review.html`)

**Tab 1: Gallery (NEW primary tab)**
- CSS grid of component group cards
- Each card: thumbnail image, OCR text overlay, confidence badge, subboard count
- Checkbox selection for batch operations
- Filter bar: status, confidence range, component class

**Tab 2: Review (existing, simplified)**
- Only entered when drilling down from gallery
- Show image detail with all lines visible at once
- Click line to enter edit mode
- Back button returns to gallery

**Tab 3: Dashboard (existing, enhanced)**
- Add group-level metrics: X/Y component groups reviewed
- Add subboard leverage metric: "Reviewed 27 groups → covered 432 instances"
- Keep existing line-level metrics

### Data Model Changes

**`line_loader.py` updates:**
- Handle raw RF-DETR JSON without `line_uid` metadata (assign automatically)
- Auto-assign chars to lines by spatial containment when no `line_uid` in description
- Parse `conf=X.XXX` from simple description format
- New: `ComponentGroup` dataclass grouping images by `region_group_id`

**`line_event_store.py` updates:**
- Add `GROUP_ACCEPTED`, `GROUP_MAJORITY_ACCEPTED`, `INSTANCE_REJECTED` event types
- Add `region_group_id` column for group-level queries

**`metrics_calculator.py` updates:**
- Group-level completion metrics
- Subboard leverage ratio

### Manifest Integration
The review app loads `crop_manifest.json` at startup to know:
- Which images are siblings (same `region_group_id`)
- Component class per image
- How many subboards per group

---

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `crop_components.py` | Crop components from board images, track subboard groups |
| `merge_detections.py` | Merge 224+448 RF-DETR, assign chars to lines, auto-accept |

### Modified Files
| File | Changes |
|------|---------|
| `run_rfdetr.py` | Fix config key, handle background class, dual-model batch mode |
| `line_loader.py` | Handle raw RF-DETR format, auto-assign chars to lines, add ComponentGroup |
| `ocr_review_app.py` | Gallery view, group-level review, subboard propagation, batch accept |
| `templates/ocr_review.html` | Gallery grid tab, subboard comparison view, batch UI |
| `line_event_store.py` | Group-level events |
| `metrics_calculator.py` | Group-level metrics, subboard leverage |
| `CLAUDE.md` | Update pipeline documentation |

### Implementation Priority
1. **`crop_components.py`** — unblocks data flow
2. **Fix `run_rfdetr.py`** — quick config fix
3. **`merge_detections.py`** — produces review-ready data
4. **`line_loader.py` updates** — handle new format + grouping
5. **Gallery view + group review** — biggest speedup for human reviewer
6. **Subboard comparison view** — second level of review
7. **Metrics updates** — tracking

### Verification
1. Extract a zip → `annotation_extraction.py` → verify board JSON
2. Crop → `crop_components.py` → verify crops are upright, manifest has sibling groups
3. Rotate → `rotate_img.py` → verify canonical orientation
4. Detect → `run_rfdetr.py` (both models) → verify detection JSONs have chars + line-bboxes
5. Merge → `merge_detections.py` → verify chars assigned to lines, OCR text generated, auto-acceptance
6. Review → gallery view shows groups, accepting one propagates to siblings
7. End-to-end: process one full zip file from extraction to final dataset
