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
    if source_has_embedded_segment_annotations(source_rrd):
        raise ValueError(
            "This recording already contains embedded segment annotations. "
            "Editing existing annotated recordings is not supported in v1."
        )


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
) -> Path:
    preview_path = create_preview_path()
    _write_annotated_rrd(source_rrd, segments, preview_path)
    cleanup_preview_file(previous_preview)
    return preview_path


def _write_annotated_rrd(
    source_rrd: Path,
    segments: Sequence[SegmentAnnotation],
    output_path: Path,
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
