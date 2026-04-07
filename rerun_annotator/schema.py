from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun import recording

ANNOTATION_SEGMENTS_PATH = "annotations/segments"
ANNOTATION_SUMMARY_PATH = "annotations/summary"
ANNOTATION_BOUNDARIES_PATH = "annotations/boundaries"

ANNOTATION_SEGMENTS_ENTITY = f"/{ANNOTATION_SEGMENTS_PATH}"
ANNOTATION_SUMMARY_ENTITY = f"/{ANNOTATION_SUMMARY_PATH}"
ANNOTATION_BOUNDARIES_ENTITY = f"/{ANNOTATION_BOUNDARIES_PATH}"

ANNOTATION_ROW_TIMELINE = "annotation_row"
ALLOWED_OUTCOMES = {"success", "fail"}
TIME_EPSILON = 1e-9
TimelineKind = Literal["sequence", "duration", "timestamp"]


@dataclass(frozen=True)
class SegmentAnnotation:
    segment_id: int
    subtask: str
    outcome: str
    timeline: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class TrimRegion:
    start_time: float
    end_time: float


@dataclass(frozen=True)
class BoundaryLogEntry:
    text: str
    level: str


def build_annotated_rrd_path(source_rrd: Path) -> Path:
    return source_rrd.with_name(f"{source_rrd.stem}.annotated{source_rrd.suffix}")


def build_output_rrd_path(source_path: Path) -> Path:
    if source_path.is_dir():
        return source_path.parent / f"{source_path.name}.annotated.rrd"
    if source_path.suffix.lower() != ".rrd":
        return source_path.parent / f"{source_path.stem}.annotated.rrd"
    return build_annotated_rrd_path(source_path)


def renumber_segments(segments: Sequence[SegmentAnnotation]) -> list[SegmentAnnotation]:
    return [replace(segment, segment_id=index) for index, segment in enumerate(segments, start=1)]


def segment_table_rows(segments: Sequence[SegmentAnnotation]) -> list[list[str | int | float]]:
    return [
        [
            segment.segment_id,
            segment.subtask,
            segment.outcome,
            segment.timeline,
            round(segment.start_time, 6),
            round(segment.end_time, 6),
        ]
        for segment in segments
    ]


def boundary_table_rows(segments: Sequence[SegmentAnnotation]) -> list[list[str | int | float]]:
    rows: list[list[str | int | float]] = []
    for segment in segments:
        rows.append(
            [
                f"{segment.segment_id}:start",
                "start",
                segment.segment_id,
                segment.timeline,
                round(segment.start_time, 6),
                segment.subtask,
                segment.outcome,
            ]
        )
        rows.append(
            [
                f"{segment.segment_id}:end",
                "end",
                segment.segment_id,
                segment.timeline,
                round(segment.end_time, 6),
                segment.subtask,
                segment.outcome,
            ]
        )
    return rows


def segment_selector_choices(segments: Sequence[SegmentAnnotation]) -> list[tuple[str, str]]:
    return [
        (
            f"#{segment.segment_id} | {segment.subtask} | {segment.start_time:.3f} -> {segment.end_time:.3f}",
            str(segment.segment_id),
        )
        for segment in segments
    ]


def build_summary_markdown(segments: Sequence[SegmentAnnotation], warnings: Sequence[str]) -> str:
    if not segments:
        lines = [
            "# Trajectory Segments",
            "",
            "_No segments annotated yet._",
        ]
    else:
        lines = [
            "# Trajectory Segments",
            "",
            "| ID | Subtask | Outcome | Timeline | Start | End |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        for segment in segments:
            lines.append(
                "| "
                f"{segment.segment_id} | "
                f"{segment.subtask} | "
                f"{segment.outcome} | "
                f"{segment.timeline} | "
                f"{segment.start_time:.6f} | "
                f"{segment.end_time:.6f} |"
            )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines)


def build_boundary_logs(segments: Sequence[SegmentAnnotation]) -> list[BoundaryLogEntry]:
    entries: list[BoundaryLogEntry] = []
    for segment in segments:
        entries.append(
            BoundaryLogEntry(
                text=(
                    f"{segment.timeline} @ {segment.start_time:.6f} | "
                    f"segment {segment.segment_id} start | {segment.subtask} | {segment.outcome}"
                ),
                level="INFO",
            )
        )
        entries.append(
            BoundaryLogEntry(
                text=(
                    f"{segment.timeline} @ {segment.end_time:.6f} | "
                    f"segment {segment.segment_id} end | {segment.subtask} | {segment.outcome}"
                ),
                level="INFO",
            )
        )
    return entries


def infer_timeline_kind(timeline: str, values: Sequence[float]) -> TimelineKind:
    timeline_name = timeline.strip().lower()

    if any(token in timeline_name for token in ("timestamp", "wall_time", "capture_time", "utc", "datetime")):
        return "timestamp"

    if any(token in timeline_name for token in ("step", "frame", "iteration", "index", "tick")) and "time" not in timeline_name:
        return "sequence"

    if any(
        token in timeline_name
        for token in ("time", "duration", "elapsed", "episode", "clock", "seconds", "secs", "milliseconds", "ms", "ns")
    ):
        return "duration"

    if values and all(abs(value - round(value)) <= 1e-6 for value in values) and max(abs(value) for value in values) < 1e7:
        return "sequence"

    return "duration"


def build_time_column(timeline: str, values: Sequence[float], kind: TimelineKind) -> rr.TimeColumn:
    if kind == "sequence":
        return rr.TimeColumn(timeline, sequence=[int(round(value)) for value in values])
    if kind == "duration":
        return rr.TimeColumn(timeline, duration=np.array([int(round(value)) for value in values], dtype="timedelta64[ns]"))
    return rr.TimeColumn(timeline, timestamp=np.array([int(round(value)) for value in values], dtype="datetime64[ns]"))


def validate_segments(segments: Sequence[SegmentAnnotation]) -> list[str]:
    warnings: list[str] = []
    previous: SegmentAnnotation | None = None

    for segment in segments:
        if segment.segment_id <= 0:
            raise ValueError("Segment IDs must be positive integers.")
        if not segment.subtask.strip():
            raise ValueError("Subtask name cannot be empty.")
        if segment.outcome not in ALLOWED_OUTCOMES:
            raise ValueError("Outcome must be either 'success' or 'fail'.")
        if not segment.timeline.strip():
            raise ValueError("Segment timeline cannot be empty.")
        if not math.isfinite(segment.start_time) or not math.isfinite(segment.end_time):
            raise ValueError("Segment times must be finite numbers.")
        if segment.start_time >= segment.end_time - TIME_EPSILON:
            raise ValueError(
                f"Segment #{segment.segment_id} must satisfy start_time < end_time."
            )

        if previous is not None:
            if segment.timeline != previous.timeline:
                raise ValueError("All segments must use the same viewer timeline.")
            if segment.start_time < previous.start_time - TIME_EPSILON:
                raise ValueError("Segments must remain in chronological order.")
            if segment.start_time < previous.end_time - TIME_EPSILON:
                raise ValueError("Segments cannot overlap.")

        previous = segment

    return warnings


def build_segment_annotation(
    segment_id: int,
    timeline: str,
    start_time: float | int | None,
    end_time: float | int | None,
    subtask: str,
    outcome: str,
) -> SegmentAnnotation:
    if start_time is None or end_time is None:
        raise ValueError("Both start and end times are required.")

    timeline_value = timeline.strip()
    if not timeline_value:
        raise ValueError("Set the segment timeline from the viewer before adding a segment.")

    subtask_value = subtask.strip()
    if not subtask_value:
        raise ValueError("Subtask name cannot be empty.")

    return SegmentAnnotation(
        segment_id=segment_id,
        subtask=subtask_value,
        outcome=outcome,
        timeline=timeline_value,
        start_time=float(start_time),
        end_time=float(end_time),
    )


def source_has_embedded_segment_annotations(source_rrd: Path) -> bool:
    schema_text = str(recording.load_recording(source_rrd).schema())
    return f"{ANNOTATION_SEGMENTS_ENTITY}:segment_id" in schema_text


def ensure_source_can_be_annotated(source_rrd: Path) -> None:
    if source_rrd.suffix.lower() != ".rrd":
        raise ValueError("Source file must be an .rrd recording.")
    if not source_rrd.exists():
        raise ValueError(f"RRD file does not exist: {source_rrd}")


def extract_segments_from_rrd(source_rrd: Path) -> list[SegmentAnnotation]:
    """Read embedded segment annotations from an annotated .rrd file."""
    source_recording = recording.load_recording(source_rrd)
    for chunk in source_recording.chunks():
        if chunk.entity_path != ANNOTATION_SEGMENTS_ENTITY:
            continue
        rb = chunk.to_record_batch()
        segments: list[SegmentAnnotation] = []
        for i in range(rb.num_rows):
            segments.append(
                SegmentAnnotation(
                    segment_id=rb.column("segment_id")[i].as_py()[0],
                    subtask=rb.column("subtask")[i].as_py()[0],
                    outcome=rb.column("outcome")[i].as_py()[0],
                    timeline=rb.column("timeline")[i].as_py()[0],
                    start_time=rb.column("start_time")[i].as_py()[0],
                    end_time=rb.column("end_time")[i].as_py()[0],
                )
            )
        return segments
    return []


def strip_annotations_to_rrd(source_rrd: Path) -> Path:
    """Create a copy of the recording with all annotation entities removed."""
    source_recording = recording.load_recording(source_rrd)
    application_id = source_recording.application_id() or "rerun_segment_annotator"
    annotation_prefixes = (ANNOTATION_SEGMENTS_ENTITY, ANNOTATION_BOUNDARIES_ENTITY, ANNOTATION_SUMMARY_ENTITY)

    stripped_path = create_materialized_source_path()
    rec = rr.RecordingStream(application_id=application_id)
    rec.save(stripped_path)

    clean_chunks = [
        chunk for chunk in source_recording.chunks()
        if not any(chunk.entity_path.startswith(p) for p in annotation_prefixes)
    ]
    clean_recording = recording.Recording.from_chunks(
        clean_chunks, application_id, source_recording.recording_id(),
    )
    rr.send_recording(clean_recording, recording=rec)
    rec.flush()
    return stripped_path


def create_materialized_source_path() -> Path:
    with tempfile.NamedTemporaryFile(prefix="rerun_materialized_source_", suffix=".rrd", delete=False) as handle:
        return Path(handle.name)


def create_preview_path() -> Path:
    with tempfile.NamedTemporaryFile(prefix="rerun_segment_preview_", suffix=".rrd", delete=False) as handle:
        return Path(handle.name)


def cleanup_temp_rrd(path_like: str | Path | None) -> None:
    if path_like is None:
        return

    path = Path(path_like)
    if path.exists() and path.name.startswith(("rerun_segment_preview_", "rerun_materialized_source_")):
        path.unlink()


def cleanup_preview_file(preview_path: str | Path | None) -> None:
    cleanup_temp_rrd(preview_path)


def materialize_source_recording(source_path: Path, previous_materialized: str | Path | None = None) -> Path:
    if source_path.is_file() and source_path.suffix.lower() == ".rrd":
        ensure_source_can_be_annotated(source_path)
        cleanup_temp_rrd(previous_materialized)
        return source_path

    if not source_path.exists():
        raise ValueError(f"Source path does not exist: {source_path}")

    materialized_path = create_materialized_source_path()
    try:
        rec = rr.RecordingStream(application_id="rerun_segment_source_materializer")
        rec.save(materialized_path)
        rec.log_file_from_path(source_path)
        rec.flush()
    except Exception:
        cleanup_temp_rrd(materialized_path)
        raise

    cleanup_temp_rrd(previous_materialized)
    return materialized_path


def save_annotated_rrd(
    source_rrd: Path,
    segments: Sequence[SegmentAnnotation],
    output_path: Path | None = None,
) -> Path:
    output_path = output_path or build_annotated_rrd_path(source_rrd)
    _write_annotated_rrd(source_rrd, segments, output_path)
    return output_path


def write_preview_rrd(
    source_rrd: Path,
    segments: Sequence[SegmentAnnotation],
    previous_preview: str | Path | None,
    *,
    blueprint_file: Path | None = None,
) -> Path:
    preview_path = create_preview_path()
    _write_annotated_rrd(
        source_rrd, segments, preview_path, skip_blueprint=blueprint_file is not None,
    )
    if blueprint_file is not None:
        _merge_blueprint(preview_path, blueprint_file)
    cleanup_preview_file(previous_preview)
    return preview_path


def _merge_blueprint(rrd_path: Path, blueprint_path: Path) -> None:
    """Re-stamp a .rbl blueprint to match the recording's application_id, then merge."""
    import shutil
    import subprocess
    import sys
    import tempfile

    source_recording = recording.load_recording(rrd_path)
    app_id = source_recording.application_id() or "rerun_segment_annotator"

    rerun_bin = shutil.which("rerun") or str(Path(sys.executable).parent / "rerun")

    # Re-stamp blueprint so its application_id matches the recording
    restamped = Path(tempfile.mktemp(suffix=".rbl"))
    try:
        subprocess.run(
            [rerun_bin, "rrd", "route", "--application-id", app_id,
             str(blueprint_path), "-o", str(restamped)],
            check=True,
            capture_output=True,
        )
        # Merge re-stamped blueprint into the RRD
        merged = rrd_path.with_suffix(".merged.rrd")
        subprocess.run(
            [rerun_bin, "rrd", "merge", str(rrd_path), str(restamped), "-o", str(merged)],
            check=True,
            capture_output=True,
        )
        merged.replace(rrd_path)
    finally:
        restamped.unlink(missing_ok=True)


def _write_annotated_rrd(
    source_rrd: Path,
    segments: Sequence[SegmentAnnotation],
    output_path: Path,
    *,
    skip_blueprint: bool = False,
) -> None:
    warnings = validate_segments(segments)
    source_recording = recording.load_recording(source_rrd)
    application_id = source_recording.application_id() or "rerun_segment_annotator"

    if output_path.exists():
        output_path.unlink()

    rec = rr.RecordingStream(application_id=application_id)
    rec.save(output_path)
    rr.send_recording(source_recording, recording=rec)

    _log_annotation_entities(rec, segments, warnings, source_recording=source_recording)
    if not skip_blueprint:
        rec.send_blueprint(build_annotation_blueprint())
    rec.flush()


def _log_annotation_entities(
    rec: rr.RecordingStream,
    segments: Sequence[SegmentAnnotation],
    warnings: Sequence[str],
    *,
    source_recording,
) -> None:
    summary_markdown = build_summary_markdown(segments, warnings)
    rec.log(
        ANNOTATION_SUMMARY_PATH,
        rr.TextDocument(summary_markdown, media_type="text/markdown"),
        static=True,
    )

    if not segments:
        rec.reset_time()
        return

    source_timeline = segments[0].timeline
    resolved_timeline_kind = get_timeline_kind(source_recording, source_timeline)

    boundary_times: list[float] = []
    boundary_segment_ids: list[int] = []
    boundary_kinds: list[str] = []
    boundary_subtasks: list[str] = []
    boundary_outcomes: list[str] = []
    boundary_timelines: list[str] = []
    boundary_logs = build_boundary_logs(segments)

    for segment in segments:
        boundary_times.extend([segment.start_time, segment.end_time])
        boundary_segment_ids.extend([segment.segment_id, segment.segment_id])
        boundary_kinds.extend(["start", "end"])
        boundary_subtasks.extend([segment.subtask, segment.subtask])
        boundary_outcomes.extend([segment.outcome, segment.outcome])
        boundary_timelines.extend([segment.timeline, segment.timeline])

    rec.send_columns(
        ANNOTATION_BOUNDARIES_PATH,
        indexes=[
            rr.TimeColumn(ANNOTATION_ROW_TIMELINE, sequence=list(range(len(boundary_times)))),
            build_time_column(source_timeline, boundary_times, resolved_timeline_kind),
        ],
        columns=rr.AnyValues.columns(
            boundary_kind=boundary_kinds,
            segment_id=boundary_segment_ids,
            subtask=boundary_subtasks,
            outcome=boundary_outcomes,
            timeline=boundary_timelines,
            boundary_time=boundary_times,
        ),
    )

    for row_index, boundary in enumerate(boundary_logs):
        rec.reset_time()
        rec.set_time(ANNOTATION_ROW_TIMELINE, sequence=row_index)
        rec.send_columns(
            ANNOTATION_BOUNDARIES_PATH,
            indexes=[
                rr.TimeColumn(ANNOTATION_ROW_TIMELINE, sequence=[row_index]),
                build_time_column(source_timeline, [boundary_times[row_index]], resolved_timeline_kind),
            ],
            columns=rr.TextLog.columns(text=[boundary.text], level=[boundary.level]),
        )

    rec.reset_time()

    rec.send_columns(
        ANNOTATION_SEGMENTS_PATH,
        indexes=[
            rr.TimeColumn(ANNOTATION_ROW_TIMELINE, sequence=list(range(len(segments)))),
            build_time_column(source_timeline, [segment.start_time for segment in segments], resolved_timeline_kind),
        ],
        columns=rr.AnyValues.columns(
            segment_id=[segment.segment_id for segment in segments],
            subtask=[segment.subtask for segment in segments],
            outcome=[segment.outcome for segment in segments],
            timeline=[segment.timeline for segment in segments],
            start_time=[segment.start_time for segment in segments],
            end_time=[segment.end_time for segment in segments],
        ),
    )


def build_annotation_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Tabs(
                rrb.Spatial2DView(origin="/", name="Spatial 2D"),
                rrb.Spatial3DView(origin="/", name="Spatial 3D"),
                rrb.TimeSeriesView(origin="/", name="Time Series"),
            ),
            rrb.Vertical(
                rrb.TextDocumentView(origin=ANNOTATION_SUMMARY_PATH, name="Segment Summary"),
                rrb.DataframeView(origin=ANNOTATION_SEGMENTS_PATH, name="Segment Rows"),
                rrb.TextLogView(origin=ANNOTATION_BOUNDARIES_PATH, name="Boundaries"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=False,
    )


def get_timeline_kind(source_recording, timeline: str) -> TimelineKind:
    arrow_schema = pa.schema(source_recording.schema())
    field = arrow_schema.field(timeline)

    if pa.types.is_timestamp(field.type):
        return "timestamp"
    if pa.types.is_duration(field.type):
        return "duration"
    if pa.types.is_integer(field.type):
        return "sequence"

    raise ValueError(f"Unsupported index type for timeline '{timeline}': {field.type}")


# ---------------------------------------------------------------------------
# Trim RRD
# ---------------------------------------------------------------------------


def trim_region_table_rows(regions: Sequence[TrimRegion]) -> list[list[int | float]]:
    return [[i + 1, round(r.start_time, 6), round(r.end_time, 6)] for i, r in enumerate(regions)]


def _ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _time_to_cli_arg(t: float) -> str:
    """Format a time value for the rerun CLI (integer for sequences, float otherwise)."""
    if abs(t - round(t)) < 1e-6:
        return str(int(round(t)))
    return str(t)


def _sort_split_files(files: list[Path]) -> list[Path]:
    """Sort split output files by (start, end) extracted from the filename.

    The CLI names boundary files with start==end (e.g. ``0__0.rrd`` for
    ``[-inf, 0)`` and ``91__91.rrd`` for ``[91, +inf)``).  Sorting by end
    as a tiebreaker ensures boundary files sort before/after data files
    at the same start time.
    """
    import re

    def extract_key(path: Path) -> tuple[float, float]:
        m = re.search(r"_(-?\d+(?:\.\d+)?)__(-?\d+(?:\.\d+)?)\.rrd$", path.name)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        return (0.0, 0.0)

    return sorted(files, key=extract_key)


def filter_segments_after_trim(
    segments: Sequence[SegmentAnnotation],
    regions: Sequence[TrimRegion],
    mode: str,
) -> list[SegmentAnnotation]:
    """Keep only segments that survive trimming."""
    result: list[SegmentAnnotation] = []
    for s in segments:
        if mode == "keep":
            if any(s.start_time >= r.start_time and s.end_time <= r.end_time for r in regions):
                result.append(s)
        else:
            if not any(_ranges_overlap(s.start_time, s.end_time, r.start_time, r.end_time) for r in regions):
                result.append(s)
    return renumber_segments(result)


def trim_rrd(
    source_rrd: Path,
    timeline: str,
    regions: Sequence[TrimRegion],
    mode: Literal["keep", "remove"],
) -> Path:
    """Trim an RRD using ``rerun rrd split`` + ``rerun rrd merge``.

    This delegates to the Rerun CLI for row-level, cross-stream splitting so
    that all entity streams are trimmed uniformly.
    """
    import re
    import shutil
    import subprocess
    import sys

    if not regions:
        raise ValueError("At least one trim region is required.")

    rerun_bin = shutil.which("rerun") or str(Path(sys.executable).parent / "rerun")

    # Scan the source recording for all unique timeline values so we can
    # compute split boundaries that land on actual data points.
    import bisect
    source_recording = recording.load_recording(source_rrd)
    timeline_values: set[float] = set()
    for chunk in source_recording.chunks():
        rb = chunk.to_record_batch()
        if timeline in rb.schema.names:
            timeline_values.update(
                float(v) for v in rb.column(timeline).to_pylist() if v is not None
            )
    sorted_tv = sorted(timeline_values)

    # Region boundaries are inclusive: [start, end].  The CLI split at time T
    # places data at exactly T into [T, next).  To make the end inclusive we
    # split at the data point AFTER end_time instead of at end_time itself.
    # This avoids carry-over artefacts (splitting between data points can
    # duplicate the preceding row into both files).
    end_bump: dict[float, float] = {}
    for r in regions:
        idx = bisect.bisect_right(sorted_tv, r.end_time)
        if idx < len(sorted_tv):
            end_bump[r.end_time] = sorted_tv[idx]
        # else: end_time is at or beyond the last data point — no bump needed
        # (the split at end_time already puts end_time in the right file and
        #  the tail file is handled by the inclusive boundary check below).

    all_times = sorted(set(
        t
        for r in regions
        for t in [r.start_time, end_bump.get(r.end_time, r.end_time)]
    ))

    with tempfile.TemporaryDirectory() as split_dir:
        # --- 1. Split --------------------------------------------------------
        cmd = [
            rerun_bin, "rrd", "split",
            "--timeline", timeline,
            "--output-dir", split_dir,
        ]
        for t in all_times:
            cmd.extend(["--time", _time_to_cli_arg(t)])
        cmd.append(str(source_rrd))
        subprocess.run(cmd, check=True, capture_output=True)

        split_files = _sort_split_files(list(Path(split_dir).glob("*.rrd")))
        if not split_files:
            raise ValueError("rerun rrd split produced no output files.")

        # --- 2. Decide which splits to keep -----------------------------------
        # For n split-times the CLI produces n+1 files.
        # File i covers [all_times[i-1], all_times[i]).
        # File 0 covers [-inf, all_times[0]) and file n covers [all_times[-1], +inf).
        keep_indices: set[int] = set()
        for i in range(len(split_files)):
            split_start = all_times[i - 1] if i > 0 else float("-inf")
            split_end = all_times[i] if i < len(all_times) else float("+inf")

            # Check against the bumped end for each region
            inside_region = any(
                split_start >= r.start_time
                and split_end <= end_bump.get(r.end_time, r.end_time)
                for r in regions
            )

            if (mode == "keep" and inside_region) or (mode == "remove" and not inside_region):
                keep_indices.add(i)

        # Handle the tail file when end_time is the last data point
        # (no bump was possible — check if the tail file only has data at end_time).
        for r in regions:
            if r.end_time in end_bump:
                continue  # already handled by the bump
            # Find the split file that starts at r.end_time
            if r.end_time in all_times:
                tail_idx = all_times.index(r.end_time) + 1
            else:
                continue
            if tail_idx >= len(split_files):
                continue
            # Check if the tail file only contains data at end_time
            tail_rec = recording.load_recording(split_files[tail_idx])
            has_later_data = False
            for chunk in tail_rec.chunks():
                rb = chunk.to_record_batch()
                if timeline in rb.schema.names:
                    times = [float(v) for v in rb.column(timeline).to_pylist() if v is not None]
                    if any(t > r.end_time for t in times):
                        has_later_data = True
                        break
            if not has_later_data:
                if mode == "keep":
                    keep_indices.add(tail_idx)
                else:
                    keep_indices.discard(tail_idx)

        kept_files = [split_files[i] for i in sorted(keep_indices)]
        if not kept_files:
            raise ValueError("No data would remain after trimming.")

        # --- 3. Merge kept splits into a single recording ----------------------
        # Each split has a unique recording ID, so we re-log all chunks
        # into one RecordingStream to produce a single unified recording.
        app_id = source_recording.application_id() or "rerun_segment_annotator"

        trimmed_path = create_materialized_source_path()
        rec = rr.RecordingStream(application_id=app_id)
        rec.save(trimmed_path)
        for f in kept_files:
            split_rec = recording.load_recording(f)
            rr.send_recording(split_rec, recording=rec)
        rec.flush()

    return trimmed_path
