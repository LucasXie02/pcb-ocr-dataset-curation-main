#!/usr/bin/env python3
"""
Line Loader
===========
Load and parse line-level annotations from LabelMe JSON files.

This module handles:
- Parsing line_uid from shape descriptions
- Grouping character bboxes by line_uid
- Extracting OCR text and metadata
- Resolving image paths

Data Model (from README):
- Every shape has unique group_id
- Line membership via line_uid in description field
- Line bbox description: "line_uid=<...>;src=<...>;conf=<...>"
- Char bbox description: "line_uid=<...>;idx=<i>;src=<...>;conf=<...>"
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BBox:
    """Bounding box in LabelMe format"""
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_points(cls, points: List[List[float]]) -> 'BBox':
        """Create from LabelMe points [[x1,y1], [x2,y2]]"""
        if len(points) != 2:
            raise ValueError(f"Expected 2 points for rectangle, got {len(points)}")
        x1, y1 = points[0]
        x2, y2 = points[1]
        return cls(
            x1=min(x1, x2),
            y1=min(y1, y2),
            x2=max(x1, x2),
            y2=max(y1, y2)
        )

    def to_points(self) -> List[List[float]]:
        """Convert to LabelMe points format"""
        return [[self.x1, self.y1], [self.x2, self.y2]]

    def area(self) -> float:
        """Compute box area"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class CharAnnotation:
    """Character-level annotation"""
    label: str
    bbox: BBox
    group_id: int
    line_uid: str
    idx: int  # Position in reading order
    src: str  # Source (hunyuan, etc.)
    conf: float  # Confidence


@dataclass
class LineAnnotation:
    """Line-level annotation with associated characters"""
    line_uid: str
    image_id: str
    bbox: BBox
    group_id: int
    src: str
    conf: float
    ocr_text: str  # From line description
    det_text: str  # Reconstructed from char labels
    chars: List[CharAnnotation] = field(default_factory=list)
    status: str = "proposed"  # proposed, accepted, uncertain
    reasons: List[str] = field(default_factory=list)  # Review reasons
    needs_review: bool = False

    def get_char_count(self) -> int:
        """Get number of detected characters"""
        return len(self.chars)

    def get_ocr_length(self) -> int:
        """Get OCR text length"""
        return len(self.ocr_text)


@dataclass
class ImageAnnotation:
    """Complete annotation for one image"""
    image_id: str
    image_path: str
    json_path: str
    image_width: int
    image_height: int
    lines: List[LineAnnotation] = field(default_factory=list)

    def get_line_count(self) -> int:
        """Get number of lines in this image"""
        return len(self.lines)

    def get_line_by_uid(self, line_uid: str) -> Optional[LineAnnotation]:
        """Get specific line by UID"""
        for line in self.lines:
            if line.line_uid == line_uid:
                return line
        return None


@dataclass
class ComponentGroup:
    """Group of sibling component instances across subboards"""
    region_group_id: str
    component_class: str
    instances: List[ImageAnnotation] = field(default_factory=list)
    majority_text: Dict[int, str] = field(default_factory=dict)  # line_index -> majority OCR text
    confidence: float = 0.0
    agreement_count: int = 0
    total_count: int = 0


# ============================================================================
# Helper Functions
# ============================================================================

def is_char_label(label: str) -> bool:
    """Check if label is a valid character (0-9, A-Z)."""
    return len(label) == 1 and label in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def parse_rfdetr_description(description: str) -> float:
    """Parse confidence from RF-DETR style description (plain float)."""
    try:
        return float(description.strip())
    except (ValueError, TypeError):
        return 0.0


def auto_assign_chars_to_lines(
    line_shapes: List[Dict],
    char_shapes: List[Dict],
    image_id: str,
) -> Tuple[List[LineAnnotation], List[CharAnnotation]]:
    """
    Assign characters to lines by spatial containment when no line_uid metadata.

    For each character, find which line bbox contains its center point.
    Sort chars left-to-right within each line.
    Set line_uid, idx, and build det_text.

    Args:
        line_shapes: Raw LabelMe shape dicts for line-bbox labels
        char_shapes: Raw LabelMe shape dicts for character labels
        image_id: Image identifier for generating line_uids

    Returns:
        (lines, unassigned_chars) — list of LineAnnotation and any chars
        that could not be assigned to a line.
    """
    # Build line bboxes
    line_bboxes: List[Tuple[int, BBox]] = []  # (index, bbox)
    for i, ls in enumerate(line_shapes):
        points = ls.get('points', [])
        try:
            bbox = BBox.from_points(points)
            line_bboxes.append((i, bbox))
        except ValueError:
            continue

    # Map line_index -> list of (cx, char_shape)
    line_char_map: Dict[int, List[Tuple[float, Dict]]] = {i: [] for i, _ in line_bboxes}
    unassigned: List[Dict] = []

    for cs in char_shapes:
        label = cs.get('label', '')
        if not is_char_label(label):
            continue
        points = cs.get('points', [])
        try:
            cb = BBox.from_points(points)
        except ValueError:
            continue
        cx = (cb.x1 + cb.x2) / 2.0
        cy = (cb.y1 + cb.y2) / 2.0

        assigned = False
        for li, lb in line_bboxes:
            if lb.x1 <= cx <= lb.x2 and lb.y1 <= cy <= lb.y2:
                line_char_map[li].append((cx, cs))
                assigned = True
                break
        if not assigned:
            unassigned.append(cs)

    # Build LineAnnotation objects
    lines: List[LineAnnotation] = []
    all_unassigned_chars: List[CharAnnotation] = []

    for li, lb in line_bboxes:
        ls = line_shapes[li]
        line_uid = f"{image_id}#L{li}"
        description = ls.get('description', '')
        conf = parse_rfdetr_description(description)
        group_id = ls.get('group_id', -1)

        # Sort chars left-to-right
        sorted_chars = sorted(line_char_map[li], key=lambda t: t[0])

        char_annotations: List[CharAnnotation] = []
        for idx, (_, cs) in enumerate(sorted_chars):
            cp = cs.get('points', [])
            try:
                cb = BBox.from_points(cp)
            except ValueError:
                continue
            cdesc = cs.get('description', '')
            cconf = parse_rfdetr_description(cdesc)
            char_annotations.append(CharAnnotation(
                label=cs.get('label', ''),
                bbox=cb,
                group_id=cs.get('group_id', -1),
                line_uid=line_uid,
                idx=idx,
                src='rfdetr',
                conf=cconf,
            ))

        det_text = ''.join(c.label for c in char_annotations)

        line = LineAnnotation(
            line_uid=line_uid,
            image_id=image_id,
            bbox=lb,
            group_id=group_id,
            src='rfdetr',
            conf=conf,
            ocr_text='',
            det_text=det_text,
            chars=char_annotations,
            status='proposed',
            reasons=[],
            needs_review=False,
        )
        lines.append(line)

    # Build CharAnnotation for unassigned chars
    for cs in unassigned:
        cp = cs.get('points', [])
        try:
            cb = BBox.from_points(cp)
        except ValueError:
            continue
        cdesc = cs.get('description', '')
        cconf = parse_rfdetr_description(cdesc)
        all_unassigned_chars.append(CharAnnotation(
            label=cs.get('label', ''),
            bbox=cb,
            group_id=cs.get('group_id', -1),
            line_uid='',
            idx=0,
            src='rfdetr',
            conf=cconf,
        ))

    return lines, all_unassigned_chars


# ============================================================================
# Description Parsing
# ============================================================================

def parse_description(description: str) -> Dict[str, str]:
    """
    Parse semicolon-separated key=value pairs from description field.

    Format: "line_uid=img001#L0;src=hunyuan;conf=0.95"

    Returns:
        Dictionary of parsed key-value pairs
    """
    result = {}
    if not description:
        return result

    # Split by semicolon
    parts = description.split(';')
    for part in parts:
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            result[key.strip()] = value.strip()

    return result


def parse_line_description(description: str) -> Tuple[str, str, float, str, bool, List[str]]:
    """
    Parse line bbox description.

    Format: "line_uid=<...>;src=<...>;conf=<...>;needs_review=1;reason=<...>"

    Returns:
        (line_uid, src, conf, ocr_text, needs_review, reasons)
    """
    parsed = parse_description(description)

    line_uid = parsed.get('line_uid', '')
    src = parsed.get('src', 'unknown')
    conf = float(parsed.get('conf', '0.0'))
    ocr_text = parsed.get('ocr', '')  # OCR text might be in description
    needs_review = parsed.get('needs_review', '0') == '1'

    # Parse reasons (comma-separated)
    reason_str = parsed.get('reason', '')
    reasons = [r.strip() for r in reason_str.split(',') if r.strip()] if reason_str else []

    return line_uid, src, conf, ocr_text, needs_review, reasons


def parse_char_description(description: str) -> Tuple[str, int, str, float]:
    """
    Parse character bbox description.

    Format: "line_uid=<...>;idx=<i>;src=<...>;conf=<...>"

    Returns:
        (line_uid, idx, src, conf)
    """
    parsed = parse_description(description)

    line_uid = parsed.get('line_uid', '')
    idx = int(parsed.get('idx', '0'))
    src = parsed.get('src', 'unknown')
    conf = float(parsed.get('conf', '0.0'))

    return line_uid, idx, src, conf


# ============================================================================
# JSON Loading
# ============================================================================

def load_labelme_json(json_path: Path) -> Optional[Dict]:
    """Load and parse LabelMe JSON file"""
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def resolve_image_path(json_path: Path, image_path_str: str) -> Optional[Path]:
    """
    Resolve image path from JSON imagePath field.

    Try in order:
    1. Absolute path
    2. Relative to JSON directory
    3. Basename in JSON directory
    """
    image_path = Path(image_path_str)

    # Try absolute
    if image_path.is_absolute() and image_path.exists():
        return image_path

    # Try relative to JSON
    json_dir = json_path.parent
    resolved = json_dir / image_path
    if resolved.exists():
        return resolved

    # Try basename only
    resolved = json_dir / image_path.name
    if resolved.exists():
        return resolved

    return None


def parse_image_annotation(json_path: Path) -> Optional[ImageAnnotation]:
    """
    Parse complete image annotation from LabelMe JSON.

    Args:
        json_path: Path to LabelMe JSON file

    Returns:
        ImageAnnotation object or None if parsing fails
    """
    data = load_labelme_json(json_path)
    if not data:
        return None

    # Extract image metadata
    image_path_str = data.get('imagePath', '')
    image_path = resolve_image_path(json_path, image_path_str)
    if not image_path:
        print(f"Warning: Could not resolve image path for {json_path}")
        image_path_str = ""
    else:
        image_path_str = str(image_path)

    image_id = json_path.stem
    image_width = data.get('imageWidth', 0)
    image_height = data.get('imageHeight', 0)

    # Parse shapes
    shapes = data.get('shapes', [])

    # Separate line bboxes and character bboxes
    line_shapes = []
    char_shapes = []

    for shape in shapes:
        label = shape.get('label', '')
        if label == 'line-bbox':
            line_shapes.append(shape)
        else:
            # Any other label is a character (0-9, A-Z)
            char_shapes.append(shape)

    # Detect format: check if any line shape has line_uid= in description
    has_line_uid = any(
        'line_uid=' in (ls.get('description', '') or '')
        for ls in line_shapes
    )

    if has_line_uid:
        # ---- Original format (line_uid in description) ----
        # Parse lines
        lines = []
        for line_shape in line_shapes:
            try:
                # Parse bbox
                points = line_shape.get('points', [])
                bbox = BBox.from_points(points)

                # Parse description
                description = line_shape.get('description', '')
                line_uid, src, conf, ocr_text, needs_review, reasons = parse_line_description(description)

                # Get group_id
                group_id = line_shape.get('group_id', -1)

                # Create line annotation
                line = LineAnnotation(
                    line_uid=line_uid,
                    image_id=image_id,
                    bbox=bbox,
                    group_id=group_id,
                    src=src,
                    conf=conf,
                    ocr_text=ocr_text,
                    det_text="",  # Will be filled from chars
                    chars=[],
                    status="uncertain" if needs_review else "proposed",
                    reasons=reasons,
                    needs_review=needs_review
                )
                lines.append(line)

            except Exception as e:
                print(f"Error parsing line shape: {e}")
                continue

        # Parse characters and assign to lines
        for char_shape in char_shapes:
            try:
                # Parse bbox
                points = char_shape.get('points', [])
                bbox = BBox.from_points(points)

                # Parse description
                description = char_shape.get('description', '')
                line_uid, idx, src, conf = parse_char_description(description)

                # Get label and group_id
                label = char_shape.get('label', '')
                group_id = char_shape.get('group_id', -1)

                # Create char annotation
                char = CharAnnotation(
                    label=label,
                    bbox=bbox,
                    group_id=group_id,
                    line_uid=line_uid,
                    idx=idx,
                    src=src,
                    conf=conf
                )

                # Find parent line and add char
                for line in lines:
                    if line.line_uid == line_uid:
                        line.chars.append(char)
                        break

            except Exception as e:
                print(f"Error parsing char shape: {e}")
                continue

        # Sort characters by index within each line and build det_text
        for line in lines:
            line.chars.sort(key=lambda c: c.idx)
            line.det_text = ''.join([c.label for c in line.chars])

    else:
        # ---- RF-DETR format (description is plain confidence float) ----
        lines, _unassigned = auto_assign_chars_to_lines(
            line_shapes, char_shapes, image_id
        )

    # Create image annotation
    img_ann = ImageAnnotation(
        image_id=image_id,
        image_path=image_path_str,
        json_path=str(json_path),
        image_width=image_width,
        image_height=image_height,
        lines=lines
    )

    return img_ann


def scan_annotation_files(root_dir: Path, pattern: str = "*.json") -> List[Path]:
    """
    Scan directory for annotation JSON files.

    Args:
        root_dir: Root directory to scan
        pattern: Glob pattern for JSON files

    Returns:
        List of JSON file paths
    """
    json_files = []
    for path in root_dir.rglob(pattern):
        if path.is_file() and not path.name.startswith('.'):
            json_files.append(path)
    return sorted(json_files)


def load_all_annotations(root_dir: Path) -> List[ImageAnnotation]:
    """
    Load all annotations from a directory.

    Args:
        root_dir: Root directory containing JSON files

    Returns:
        List of ImageAnnotation objects
    """
    json_files = scan_annotation_files(root_dir)
    annotations = []

    for json_path in json_files:
        img_ann = parse_image_annotation(json_path)
        if img_ann:
            annotations.append(img_ann)

    return annotations


def get_all_line_uids(annotations: List[ImageAnnotation]) -> List[str]:
    """
    Extract all line UIDs from annotations.

    Args:
        annotations: List of image annotations

    Returns:
        List of unique line UIDs
    """
    line_uids = []
    for img_ann in annotations:
        for line in img_ann.lines:
            if line.line_uid:
                line_uids.append(line.line_uid)
    return line_uids


# ============================================================================
# Convenience Functions
# ============================================================================

def find_line_in_annotations(line_uid: str, annotations: List[ImageAnnotation]) -> Optional[Tuple[ImageAnnotation, LineAnnotation]]:
    """
    Find a specific line across all annotations.

    Args:
        line_uid: Line unique identifier
        annotations: List of image annotations

    Returns:
        Tuple of (ImageAnnotation, LineAnnotation) or None
    """
    for img_ann in annotations:
        line = img_ann.get_line_by_uid(line_uid)
        if line:
            return img_ann, line
    return None


def build_component_groups(
    annotations: List[ImageAnnotation],
    manifest_path: Path,
) -> List[ComponentGroup]:
    """
    Build component groups from crop manifest for subboard comparison.

    The manifest is a JSON file mapping region_group_id to a list of crop
    entries.  Each crop entry has at least ``image_id`` and
    ``component_class`` fields that let us link back to *annotations*.

    For every group we compute the majority OCR text per line index
    (by simple vote across sibling instances) along with agreement stats.

    Args:
        annotations: Already-parsed image annotations (keyed by image_id).
        manifest_path: Path to the crop manifest JSON file.

    Returns:
        List of ComponentGroup objects.
    """
    if not manifest_path.exists():
        print(f"Warning: manifest not found at {manifest_path}")
        return []

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error loading manifest {manifest_path}: {e}")
        return []

    # Index annotations by image_id for fast lookup
    ann_by_id: Dict[str, ImageAnnotation] = {a.image_id: a for a in annotations}

    # Convert crop_components.py format {"board_name":..., "crops":[...]}
    # to {region_group_id: [{"image_id":..., "component_class":...}, ...]}
    if "crops" in manifest and isinstance(manifest["crops"], list):
        converted: Dict[str, list] = {}
        for entry in manifest["crops"]:
            gid = entry.get("region_group_id", "")
            if not gid:
                continue
            crop_file = entry.get("crop_file", "")
            image_id = Path(crop_file).stem if crop_file else ""
            converted.setdefault(gid, []).append({
                "image_id": image_id,
                "component_class": entry.get("component_class", ""),
            })
        manifest = converted

    groups: List[ComponentGroup] = []
    for region_group_id, entries in manifest.items():
        if not isinstance(entries, list):
            continue

        component_class = ''
        instances: List[ImageAnnotation] = []
        for entry in entries:
            img_id = entry.get('image_id', '')
            if not component_class:
                component_class = entry.get('component_class', '')
            if img_id in ann_by_id:
                instances.append(ann_by_id[img_id])

        if not instances:
            continue

        # Compute majority text per line index
        from collections import Counter
        line_texts: Dict[int, List[str]] = {}
        for inst in instances:
            for li, line in enumerate(inst.lines):
                text = line.ocr_text or line.det_text
                line_texts.setdefault(li, []).append(text)

        majority_text: Dict[int, str] = {}
        total_agree = 0
        total_lines = 0
        for li, texts in line_texts.items():
            counter = Counter(texts)
            most_common_text, most_common_count = counter.most_common(1)[0]
            majority_text[li] = most_common_text
            total_agree += most_common_count
            total_lines += len(texts)

        confidence = total_agree / total_lines if total_lines > 0 else 0.0

        groups.append(ComponentGroup(
            region_group_id=region_group_id,
            component_class=component_class,
            instances=instances,
            majority_text=majority_text,
            confidence=confidence,
            agreement_count=total_agree,
            total_count=total_lines,
        ))

    return groups


def build_text_groups(
    annotations: List[ImageAnnotation],
) -> List[ComponentGroup]:
    """
    Build component groups by OCR text similarity (for single-board / no-panel datasets).

    Groups all images whose combined OCR text is identical. This enables batch
    review when there is no subboard panel structure to exploit.

    Args:
        annotations: Already-parsed image annotations.

    Returns:
        List of ComponentGroup objects keyed by OCR text.
    """
    from collections import Counter

    # Group annotations by their combined OCR text
    text_to_anns: Dict[str, List[ImageAnnotation]] = {}
    for ann in annotations:
        texts = []
        for line in ann.lines:
            t = line.ocr_text or line.det_text or ''
            if t:
                texts.append(t)
        combined = ','.join(texts) if texts else '_EMPTY_'
        text_to_anns.setdefault(combined, []).append(ann)

    groups: List[ComponentGroup] = []
    for text_key, instances in text_to_anns.items():
        # Include singletons so all images appear in gallery

        # Compute majority text per line index and agreement stats
        line_texts: Dict[int, List[str]] = {}
        for inst in instances:
            for li, line in enumerate(inst.lines):
                t = line.ocr_text or line.det_text
                line_texts.setdefault(li, []).append(t)

        majority_text: Dict[int, str] = {}
        total_agree = 0
        total_lines = 0
        for li, texts in line_texts.items():
            counter = Counter(texts)
            most_common_text, most_common_count = counter.most_common(1)[0]
            majority_text[li] = most_common_text
            total_agree += most_common_count
            total_lines += len(texts)

        confidence = total_agree / total_lines if total_lines > 0 else 0.0

        display_text = text_key if text_key != '_EMPTY_' else '(空)'
        groups.append(ComponentGroup(
            region_group_id=f'text:{display_text}',
            component_class='',
            instances=instances,
            majority_text=majority_text,
            confidence=confidence,
            agreement_count=total_agree,
            total_count=total_lines,
        ))

    # Sort by group size descending (largest groups first for review efficiency)
    groups.sort(key=lambda g: len(g.instances), reverse=True)
    return groups
