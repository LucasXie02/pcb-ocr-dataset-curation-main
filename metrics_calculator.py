#!/usr/bin/env python3
"""
Metrics Calculator
==================
Calculate KPIs and analytics for OCR review dashboard.

Metrics from README Section 7:
- Lines proposed
- Auto-accepted lines
- Acceptance rate
- Review rate
- Manual edit rate
- Orientation override rate

Analytics:
- Funnel: raw → accepted → reviewed → exported
- Failure reason counts (LEN_MISMATCH, NO_CHARS, NO_OCR, etc.)
- Mismatch histogram: k - len(S)
- Productivity: time/line, edits/line
"""

from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from line_loader import ImageAnnotation, LineAnnotation, load_all_annotations
from line_event_store import LineEventStore, EventType


class MetricsCalculator:
    """
    Calculate metrics and KPIs for the OCR review system.

    Usage:
        calc = MetricsCalculator(event_store, annotations)

        # Get KPIs
        kpis = calc.get_kpis()

        # Get funnel data
        funnel = calc.get_funnel()

        # Get failure reasons
        reasons = calc.get_failure_reasons()
    """

    def __init__(self, event_store: LineEventStore, annotations: List[ImageAnnotation],
                 manifest=None):
        """
        Initialize metrics calculator.

        Args:
            event_store: Event store for tracking line events
            annotations: List of image annotations
            manifest: Optional crop_manifest data for group analysis
        """
        self.event_store = event_store
        self.annotations = annotations
        self.manifest = manifest

    def get_kpis(self) -> Dict[str, Any]:
        """
        Calculate Key Performance Indicators.

        Returns:
            {
                "lines_proposed": int,
                "lines_auto_accepted": int,
                "lines_uncertain": int,
                "lines_reviewed": int,
                "lines_edited": int,
                "lines_deleted": int,
                "acceptance_rate": float,
                "review_rate": float,
                "manual_edit_rate": float
            }
        """
        # Count lines by initial status (from annotations)
        lines_proposed = 0
        lines_auto_accepted = 0
        lines_uncertain = 0

        for img_ann in self.annotations:
            for line in img_ann.lines:
                lines_proposed += 1
                if line.status == "accepted" or (not line.needs_review):
                    lines_auto_accepted += 1
                else:
                    lines_uncertain += 1

        # Count lines by current status within this subdir (from events)
        lines_reviewed = 0
        lines_edited = 0
        lines_deleted = 0

        for img_ann in self.annotations:
            for line in img_ann.lines:
                line_status = self.event_store.get_line_status(line.line_uid)
                status = line_status.get('status')
                if status == 'reviewed':
                    lines_reviewed += 1
                elif status == 'edited':
                    lines_edited += 1
                elif status == 'deleted':
                    lines_deleted += 1

        # Calculate rates
        acceptance_rate = (lines_auto_accepted / lines_proposed * 100) if lines_proposed > 0 else 0
        review_rate = (lines_uncertain / lines_proposed * 100) if lines_proposed > 0 else 0
        manual_edit_rate = (lines_edited / lines_proposed * 100) if lines_proposed > 0 else 0

        return {
            "lines_proposed": lines_proposed,
            "lines_auto_accepted": lines_auto_accepted,
            "lines_uncertain": lines_uncertain,
            "lines_reviewed": lines_reviewed,
            "lines_edited": lines_edited,
            "lines_deleted": lines_deleted,
            "acceptance_rate": round(acceptance_rate, 2),
            "review_rate": round(review_rate, 2),
            "manual_edit_rate": round(manual_edit_rate, 2)
        }

    def get_funnel(self) -> Dict[str, int]:
        """
        Calculate acceptance funnel.

        Funnel stages:
        - raw: Total lines proposed
        - accepted: Auto-accepted OR reviewed/edited
        - reviewed: Lines that have been reviewed by humans
        - exported: Lines ready for training (accepted + reviewed + edited)

        Returns:
            {
                "raw": int,
                "accepted": int,
                "reviewed": int,
                "exported": int
            }
        """
        # Count lines at each stage
        raw = 0
        auto_accepted = 0
        reviewed_or_edited = 0

        for img_ann in self.annotations:
            for line in img_ann.lines:
                raw += 1

                # Check if auto-accepted
                if line.status == "accepted" or (not line.needs_review):
                    auto_accepted += 1

                # Check if reviewed/edited (from events)
                line_status = self.event_store.get_line_status(line.line_uid)
                if line_status['status'] in ['reviewed', 'edited']:
                    reviewed_or_edited += 1

        accepted = auto_accepted
        reviewed = reviewed_or_edited
        exported = auto_accepted + reviewed_or_edited  # Lines ready for use

        return {
            "raw": raw,
            "accepted": accepted,
            "reviewed": reviewed,
            "exported": exported
        }

    def get_failure_reasons(self) -> Dict[str, int]:
        """
        Count failure reasons for lines needing review.

        Reasons from README:
        - LEN_MISMATCH: k != len(S)
        - NO_CHARS: No character boxes detected
        - NO_OCR: No OCR text available
        - ORDER_AMBIGUOUS: Character order unclear
        - BOX_ASSIGNMENT_SUSPECT: Suspicious spacing

        Returns:
            Dictionary mapping reason to count
        """
        reason_counts = Counter()

        for img_ann in self.annotations:
            for line in img_ann.lines:
                if line.needs_review:
                    for reason in line.reasons:
                        reason_counts[reason] += 1

        return dict(reason_counts)

    def get_mismatch_histogram(self) -> Dict[int, int]:
        """
        Calculate histogram of (k - len(S)) for lines.

        k = number of detected character boxes
        S = OCR string

        Returns:
            Dictionary mapping difference to count
            {
                -2: 5,   # 5 lines with 2 fewer chars than OCR
                -1: 12,
                0: 245,  # 245 lines with exact match
                1: 8,
                2: 3
            }
        """
        histogram = Counter()

        for img_ann in self.annotations:
            for line in img_ann.lines:
                k = line.get_char_count()
                s_len = line.get_ocr_length()
                diff = k - s_len
                histogram[diff] += 1

        return dict(sorted(histogram.items()))

    def get_productivity_metrics(self) -> Dict[str, Any]:
        """
        Calculate productivity metrics.

        Returns:
            {
                "avg_time_per_line": float (seconds),
                "avg_edits_per_line": float,
                "total_review_time": float (seconds)
            }
        """
        # Get all line UIDs
        all_uids = self.event_store.get_all_line_uids()

        total_time = 0.0
        total_edits = 0
        lines_with_time = 0

        for uid in all_uids:
            events = self.event_store.get_line_events(uid)

            if len(events) >= 2:
                # Calculate time between first and last event
                first_time = datetime.fromisoformat(events[0].timestamp)
                last_time = datetime.fromisoformat(events[-1].timestamp)
                duration = (last_time - first_time).total_seconds()

                total_time += duration
                lines_with_time += 1

            # Count edits
            edit_count = sum(1 for e in events if e.event_type == EventType.EDITED.value)
            total_edits += edit_count

        avg_time = (total_time / lines_with_time) if lines_with_time > 0 else 0
        avg_edits = (total_edits / len(all_uids)) if len(all_uids) > 0 else 0

        return {
            "avg_time_per_line": round(avg_time, 2),
            "avg_edits_per_line": round(avg_edits, 2),
            "total_review_time": round(total_time, 2)
        }

    def get_group_metrics(self) -> Dict[str, Any]:
        """
        Calculate group-level metrics for subboard comparison review.

        Groups are identified either from the manifest (if provided) or by
        extracting a common prefix pattern from image IDs (everything before
        the last underscore-separated segment, e.g. "boardA_crop1" and
        "boardA_crop2" share the group prefix "boardA").

        Returns:
            {
                "total_groups": int,
                "groups_reviewed": int,
                "groups_pending": int,
                "total_instances": int,
                "instances_covered": int,   # instances covered by group-level review
                "leverage_ratio": float,     # instances_covered / groups_reviewed
                "groups_all_agree": int,
                "groups_with_outliers": int
            }
        """
        # --- Build group mapping: group_key -> list of line_uids -----------
        groups: Dict[str, List[str]] = defaultdict(list)

        if self.manifest is not None:
            # Manifest-based grouping: manifest maps group_key -> instance list
            for group_key, instances in self.manifest.items():
                if isinstance(instances, list):
                    for inst in instances:
                        # Instance may be a dict with image_id or a plain string
                        img_id = inst.get("image_id", inst) if isinstance(inst, dict) else str(inst)
                        for img_ann in self.annotations:
                            if img_ann.image_id == img_id:
                                for line in img_ann.lines:
                                    groups[group_key].append(line.line_uid)
        else:
            # Heuristic grouping: common prefix before last '_' segment
            for img_ann in self.annotations:
                parts = img_ann.image_id.rsplit("_", 1)
                group_key = parts[0] if len(parts) > 1 else img_ann.image_id
                for line in img_ann.lines:
                    groups[group_key].append(line.line_uid)

        total_groups = len(groups)
        total_instances = sum(len(uids) for uids in groups.values())

        # --- Classify groups using event store ----------------------------
        groups_reviewed = 0
        groups_pending = 0
        instances_covered = 0
        groups_all_agree = 0
        groups_with_outliers = 0

        group_event_types = {
            EventType.GROUP_ACCEPTED.value,
            EventType.GROUP_MAJORITY_ACCEPTED.value,
        }

        for group_key, line_uids in groups.items():
            # Check whether any line in the group has a group-level event
            has_group_event = False
            has_majority_event = False

            for uid in line_uids:
                events = self.event_store.get_line_events(uid)
                for ev in events:
                    if ev.event_type in group_event_types:
                        has_group_event = True
                    if ev.event_type == EventType.GROUP_MAJORITY_ACCEPTED.value:
                        has_majority_event = True

            if has_group_event:
                groups_reviewed += 1
                instances_covered += len(line_uids)

                if has_majority_event:
                    groups_with_outliers += 1
                else:
                    groups_all_agree += 1
            else:
                groups_pending += 1

        leverage_ratio = (instances_covered / groups_reviewed) if groups_reviewed > 0 else 0.0

        return {
            "total_groups": total_groups,
            "groups_reviewed": groups_reviewed,
            "groups_pending": groups_pending,
            "total_instances": total_instances,
            "instances_covered": instances_covered,
            "leverage_ratio": round(leverage_ratio, 2),
            "groups_all_agree": groups_all_agree,
            "groups_with_outliers": groups_with_outliers,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics in one call.

        Returns:
            {
                "kpis": {...},
                "funnel": {...},
                "failure_reasons": {...},
                "mismatch_histogram": {...},
                "productivity": {...},
                "group_metrics": {...}
            }
        """
        return {
            "kpis": self.get_kpis(),
            "funnel": self.get_funnel(),
            "failure_reasons": self.get_failure_reasons(),
            "mismatch_histogram": self.get_mismatch_histogram(),
            "productivity": self.get_productivity_metrics(),
            "group_metrics": self.get_group_metrics()
        }

    def get_lines_by_filter(self, filter_type: str, filter_value: Optional[str] = None) -> List[str]:
        """
        Get filtered list of line UIDs.

        Args:
            filter_type: Type of filter (status, reason, image)
            filter_value: Value to filter by

        Returns:
            List of line UIDs matching filter
        """
        matching_uids = []

        if filter_type == "status":
            # Filter by current status
            for img_ann in self.annotations:
                for line in img_ann.lines:
                    status = self.event_store.get_line_status(line.line_uid)
                    if status['status'] == filter_value:
                        matching_uids.append(line.line_uid)

        elif filter_type == "reason":
            # Filter by failure reason
            for img_ann in self.annotations:
                for line in img_ann.lines:
                    if filter_value in line.reasons:
                        matching_uids.append(line.line_uid)

        elif filter_type == "image":
            # Filter by image ID
            for img_ann in self.annotations:
                if img_ann.image_id == filter_value:
                    for line in img_ann.lines:
                        matching_uids.append(line.line_uid)

        elif filter_type == "needs_review":
            # Filter lines that need review
            for img_ann in self.annotations:
                for line in img_ann.lines:
                    if line.needs_review:
                        matching_uids.append(line.line_uid)

        elif filter_type == "unreviewed":
            # Filter lines that haven't been reviewed yet
            for img_ann in self.annotations:
                for line in img_ann.lines:
                    line_status = self.event_store.get_line_status(line.line_uid)
                    if line_status['status'] in ['proposed', 'uncertain', 'accepted']:
                        # Not yet reviewed by human
                        if line_status['review_count'] == 0:
                            matching_uids.append(line.line_uid)

        return matching_uids

    def generate_summary_report(self) -> str:
        """
        Generate a human-readable summary report.

        Returns:
            Formatted string with metrics summary
        """
        metrics = self.get_all_metrics()

        report = []
        report.append("=" * 60)
        report.append("OCR REVIEW METRICS SUMMARY")
        report.append("=" * 60)
        report.append("")

        # KPIs
        kpis = metrics['kpis']
        report.append("KEY PERFORMANCE INDICATORS")
        report.append("-" * 60)
        report.append(f"  Lines Proposed:        {kpis['lines_proposed']}")
        report.append(f"  Auto-Accepted:         {kpis['lines_auto_accepted']} ({kpis['acceptance_rate']:.1f}%)")
        report.append(f"  Needs Review:          {kpis['lines_uncertain']} ({kpis['review_rate']:.1f}%)")
        report.append(f"  Reviewed by Human:     {kpis['lines_reviewed']}")
        report.append(f"  Manually Edited:       {kpis['lines_edited']} ({kpis['manual_edit_rate']:.1f}%)")
        report.append(f"  Deleted:               {kpis['lines_deleted']}")
        report.append("")

        # Funnel
        funnel = metrics['funnel']
        report.append("ACCEPTANCE FUNNEL")
        report.append("-" * 60)
        report.append(f"  Raw Lines:             {funnel['raw']}")
        report.append(f"  Accepted:              {funnel['accepted']}")
        report.append(f"  Reviewed:              {funnel['reviewed']}")
        report.append(f"  Exported:              {funnel['exported']}")
        report.append("")

        # Failure Reasons
        reasons = metrics['failure_reasons']
        if reasons:
            report.append("FAILURE REASONS")
            report.append("-" * 60)
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                report.append(f"  {reason:20s}     {count}")
            report.append("")

        # Mismatch Histogram
        histogram = metrics['mismatch_histogram']
        if histogram:
            report.append("MISMATCH HISTOGRAM (k - len(S))")
            report.append("-" * 60)
            for diff, count in sorted(histogram.items()):
                bar = '#' * min(count, 50)
                report.append(f"  {diff:3d}:  {bar} ({count})")
            report.append("")

        # Productivity
        prod = metrics['productivity']
        report.append("PRODUCTIVITY METRICS")
        report.append("-" * 60)
        report.append(f"  Avg Time per Line:     {prod['avg_time_per_line']:.2f} seconds")
        report.append(f"  Avg Edits per Line:    {prod['avg_edits_per_line']:.2f}")
        report.append(f"  Total Review Time:     {prod['total_review_time']:.2f} seconds")
        report.append("")

        # Group Metrics
        gm = metrics['group_metrics']
        report.append("GROUP-LEVEL METRICS")
        report.append("-" * 60)
        report.append(f"  Total Groups:          {gm['total_groups']}")
        report.append(f"  Groups Reviewed:       {gm['groups_reviewed']}")
        report.append(f"  Groups Pending:        {gm['groups_pending']}")
        report.append(f"  Total Instances:       {gm['total_instances']}")
        report.append(f"  Instances Covered:     {gm['instances_covered']}")
        report.append(f"  Leverage Ratio:        {gm['leverage_ratio']:.2f}")
        report.append(f"  Groups All Agree:      {gm['groups_all_agree']}")
        report.append(f"  Groups With Outliers:  {gm['groups_with_outliers']}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_metrics_calculator(db_path: Path, annotations_dir: Path) -> MetricsCalculator:
    """
    Create a metrics calculator from database and annotations directory.

    Args:
        db_path: Path to event database
        annotations_dir: Directory containing LabelMe JSON files

    Returns:
        MetricsCalculator instance
    """
    from line_event_store import LineEventStore

    event_store = LineEventStore(db_path)
    annotations = load_all_annotations(annotations_dir)

    return MetricsCalculator(event_store, annotations)
