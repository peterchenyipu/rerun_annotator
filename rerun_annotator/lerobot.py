from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr

from rerun_annotator.schema import (
    SegmentAnnotation,
    cleanup_temp_rrd,
    create_materialized_source_path,
    ensure_source_can_be_annotated,
)

LEROBOT_INFO_PATH = Path("meta/info.json")
LEROBOT_TASKS_PATH = Path("meta/tasks.parquet")
LEROBOT_DATA_DIR = Path("data")
LEROBOT_VIDEOS_DIR = Path("videos")
LEROBOT_TIMELINE_FRAME = "frame_index"
LEROBOT_IGNORED_COLUMNS = {"episode_index", "index", "frame_index", "timestamp"}


@dataclass(frozen=True)
class NativeRrdSource:
    path: Path


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    row_count: int
    global_index_start: int
    global_index_end: int
    timestamp_start_s: float
    timestamp_end_s: float
    task_index: int
    task_label: str


@dataclass(frozen=True)
class VideoShard:
    path: Path
    global_index_start: int
    global_index_end: int
    frame_count: int


@dataclass(frozen=True)
class LeRobotDatasetSource:
    dataset_path: Path
    info: dict[str, Any]
    data_file: Path
    episodes: list[EpisodeRecord]
    video_streams: list[str]
    video_shards: dict[str, list[VideoShard]]


ResolvedSource = NativeRrdSource | LeRobotDatasetSource


def resolve_source(source_path: Path) -> ResolvedSource:
    if source_path.is_file() and source_path.suffix.lower() == ".rrd":
        ensure_source_can_be_annotated(source_path)
        return NativeRrdSource(path=source_path)

    if is_lerobot_dataset_directory(source_path):
        return load_lerobot_dataset_source(source_path)

    if not source_path.exists():
        raise ValueError(f"Source path does not exist: {source_path}")

    raise ValueError(
        "Unsupported source path. Use an .rrd recording or a LeRobot dataset directory containing "
        "`meta/info.json`, `data/`, and `videos/`."
    )


def is_lerobot_dataset_directory(source_path: Path) -> bool:
    return (
        source_path.is_dir()
        and (source_path / LEROBOT_INFO_PATH).exists()
        and (source_path / LEROBOT_DATA_DIR).exists()
        and (source_path / LEROBOT_VIDEOS_DIR).exists()
    )


def load_lerobot_dataset_source(dataset_path: Path) -> LeRobotDatasetSource:
    info = json.loads((dataset_path / LEROBOT_INFO_PATH).read_text())
    data_file = _find_single_data_file(dataset_path)
    task_labels = _load_task_labels(dataset_path)
    episodes = _load_episode_records(data_file, task_labels)
    video_streams = _discover_video_streams(info)
    video_shards = {stream: _build_video_shards(dataset_path, stream) for stream in video_streams}
    return LeRobotDatasetSource(
        dataset_path=dataset_path,
        info=info,
        data_file=data_file,
        episodes=episodes,
        video_streams=video_streams,
        video_shards=video_shards,
    )


def render_source_summary_markdown(source: ResolvedSource) -> str:
    if isinstance(source, NativeRrdSource):
        return (
            "### Source Summary\n"
            "- Type: `RRD`\n"
            f"- Path: `{source.path}`"
        )

    camera_list = ", ".join(_short_stream_name(stream) for stream in source.video_streams) or "none"
    task_labels = sorted({episode.task_label for episode in source.episodes if episode.task_label})
    sample_task = task_labels[0] if task_labels else ""
    return (
        "### Source Summary\n"
        "- Type: `LeRobot dataset`\n"
        f"- Dataset: `{source.dataset_path}`\n"
        f"- Robot: `{source.info.get('robot_type', 'unknown')}`\n"
        f"- FPS: `{source.info.get('fps', 'unknown')}`\n"
        f"- Episodes: `{len(source.episodes)}`\n"
        f"- Cameras: `{camera_list}`\n"
        f"- Sample task: `{sample_task or 'unknown'}`"
    )


def episode_selector_choices(source: LeRobotDatasetSource) -> list[tuple[str, str]]:
    return [
        (
            f"#{episode.episode_index} | {episode.task_label or 'task'} | "
            f"{episode.row_count} frames | {episode.timestamp_end_s:.2f}s",
            str(episode.episode_index),
        )
        for episode in source.episodes
    ]


def build_lerobot_output_rrd_path(dataset_path: Path, episode_index: int) -> Path:
    return build_lerobot_annotation_dir(dataset_path) / f"episode-{episode_index}.annotated.rrd"


def build_lerobot_manifest_path(dataset_path: Path) -> Path:
    return build_lerobot_annotation_dir(dataset_path) / "manifest.parquet"


def build_lerobot_annotation_dir(dataset_path: Path) -> Path:
    return dataset_path.parent / f"{dataset_path.name}.annotations"


def materialize_lerobot_episode(
    source: LeRobotDatasetSource,
    episode_index: int,
    previous_materialized: str | Path | None = None,
) -> Path:
    episode = get_episode_record(source, episode_index)
    table = pq.read_table(source.data_file, filters=[("episode_index", "=", episode_index)])
    if table.num_rows == 0:
        raise ValueError(f"Episode #{episode_index} does not contain any frames.")

    frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
    global_indices = [int(value) for value in table["index"].to_pylist()]

    materialized_path = create_materialized_source_path()
    try:
        rec = rr.RecordingStream(application_id=f"lerobot_episode_{episode_index}")
        rec.save(materialized_path)

        rec.log(
            "source/summary",
            rr.TextDocument(
                (
                    f"# LeRobot Episode\n\n"
                    f"- Dataset: `{source.dataset_path}`\n"
                    f"- Episode: `{episode_index}`\n"
                    f"- Task: `{episode.task_label or 'unknown'}`\n"
                    f"- Frames: `{episode.row_count}`\n"
                    f"- Duration: `{episode.timestamp_end_s:.6f}s`"
                ),
                media_type="text/markdown",
            ),
            static=True,
        )
        row_lookup = {global_index: frame_index for global_index, frame_index in zip(global_indices, frame_indices)}

        for feature_key, feature in source.info.get("features", {}).items():
            dtype = feature.get("dtype")
            if feature_key in LEROBOT_IGNORED_COLUMNS:
                continue

            if dtype == "video":
                _log_video_feature(rec, source, feature_key, episode, row_lookup)
                continue

            if dtype in {"float32", "float64"} and feature_key in table.column_names:
                _log_scalar_feature(rec, feature_key, table[feature_key], frame_indices, feature)
                continue

            if dtype == "int64" and feature_key == "task_index" and feature_key in table.column_names:
                _log_task_feature(rec, table[feature_key], frame_indices, episode)

        rec.flush()
    except Exception:
        cleanup_temp_rrd(materialized_path)
        raise

    cleanup_temp_rrd(previous_materialized)
    return materialized_path


def update_lerobot_annotation_manifest(
    source: LeRobotDatasetSource,
    episode_index: int,
    annotated_rrd_path: Path,
    segments: list[SegmentAnnotation],
) -> Path:
    output_dir = build_lerobot_annotation_dir(source.dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = build_lerobot_manifest_path(source.dataset_path)

    row = {
        "dataset_path": str(source.dataset_path),
        "episode_index": int(episode_index),
        "annotated_rrd_path": str(annotated_rrd_path),
        "source_materialization": f"episode:{episode_index}",
        "timeline": segments[0].timeline if segments else "",
        "segment_count": len(segments),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "first_subtask": segments[0].subtask if segments else "",
        "last_subtask": segments[-1].subtask if segments else "",
        "success_count": sum(1 for segment in segments if segment.outcome == "success"),
        "fail_count": sum(1 for segment in segments if segment.outcome == "fail"),
    }

    schema = pa.schema(
        [
            pa.field("dataset_path", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("annotated_rrd_path", pa.string()),
            pa.field("source_materialization", pa.string()),
            pa.field("timeline", pa.string()),
            pa.field("segment_count", pa.int64()),
            pa.field("updated_at", pa.string()),
            pa.field("first_subtask", pa.string()),
            pa.field("last_subtask", pa.string()),
            pa.field("success_count", pa.int64()),
            pa.field("fail_count", pa.int64()),
        ]
    )

    rows = []
    if manifest_path.exists():
        existing = pq.read_table(manifest_path)
        rows.extend(existing.to_pylist())

    rows = [existing_row for existing_row in rows if int(existing_row["episode_index"]) != episode_index]
    rows.append(row)
    rows.sort(key=lambda item: int(item["episode_index"]))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), manifest_path)
    return manifest_path


def get_episode_record(source: LeRobotDatasetSource, episode_index: int) -> EpisodeRecord:
    for episode in source.episodes:
        if episode.episode_index == episode_index:
            return episode
    raise ValueError(f"Episode #{episode_index} does not exist in `{source.dataset_path}`.")


def _find_single_data_file(dataset_path: Path) -> Path:
    data_files = sorted((dataset_path / LEROBOT_DATA_DIR).rglob("*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet data files found in `{dataset_path / LEROBOT_DATA_DIR}`.")
    if len(data_files) > 1:
        raise ValueError("Multiple LeRobot parquet data files are not supported in v1.")
    return data_files[0]


def _load_task_labels(dataset_path: Path) -> dict[int, str]:
    task_table = pq.read_table(dataset_path / LEROBOT_TASKS_PATH)
    task_index_values = [int(value) for value in task_table["task_index"].to_pylist()]
    task_label_column = task_table.column_names[-1]
    task_labels = [str(value) for value in task_table[task_label_column].to_pylist()]
    return dict(zip(task_index_values, task_labels))


def _load_episode_records(data_file: Path, task_labels: dict[int, str]) -> list[EpisodeRecord]:
    table = pq.read_table(data_file, columns=["episode_index", "index", "timestamp", "task_index"])
    episode_indices = [int(value) for value in table["episode_index"].to_pylist()]
    global_indices = [int(value) for value in table["index"].to_pylist()]
    timestamps = [float(value) for value in table["timestamp"].to_pylist()]
    task_indices = [int(value) for value in table["task_index"].to_pylist()]

    episodes: list[EpisodeRecord] = []
    start = 0
    while start < len(episode_indices):
        episode_index = episode_indices[start]
        end = start + 1
        while end < len(episode_indices) and episode_indices[end] == episode_index:
            end += 1

        task_index = task_indices[start]
        episodes.append(
            EpisodeRecord(
                episode_index=episode_index,
                row_count=end - start,
                global_index_start=global_indices[start],
                global_index_end=global_indices[end - 1],
                timestamp_start_s=timestamps[start],
                timestamp_end_s=timestamps[end - 1],
                task_index=task_index,
                task_label=task_labels.get(task_index, ""),
            )
        )
        start = end

    return episodes


def _discover_video_streams(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return sorted(
        feature_name
        for feature_name, feature_info in features.items()
        if isinstance(feature_info, dict) and feature_info.get("dtype") == "video"
    )


def _build_video_shards(dataset_path: Path, stream_name: str) -> list[VideoShard]:
    stream_dir = dataset_path / LEROBOT_VIDEOS_DIR / stream_name
    shard_paths = sorted(stream_dir.rglob("*.mp4"))
    shards: list[VideoShard] = []
    next_start = 0
    for shard_path in shard_paths:
        frame_count = _get_video_frame_count(shard_path)
        shards.append(
            VideoShard(
                path=shard_path,
                global_index_start=next_start,
                global_index_end=next_start + frame_count - 1,
                frame_count=frame_count,
            )
        )
        next_start += frame_count
    return shards


def _get_video_frame_count(video_path: Path) -> int:
    probe = _probe_video_stream(video_path)
    if probe is not None and probe.get("nb_frames"):
        return int(probe["nb_frames"])

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Failed to open video shard `{video_path}`.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video shard `{video_path}` did not report a positive frame count.")
        return frame_count
    finally:
        capture.release()


def _log_video_feature(
    rec: rr.RecordingStream,
    source: LeRobotDatasetSource,
    feature_key: str,
    episode: EpisodeRecord,
    row_lookup: dict[int, int],
) -> None:
    for shard_index, shard in enumerate(source.video_shards.get(feature_key, [])):
        overlap_start = max(shard.global_index_start, episode.global_index_start)
        overlap_end = min(shard.global_index_end, episode.global_index_end)
        if overlap_start > overlap_end:
            continue

        local_start = overlap_start - shard.global_index_start
        local_end = overlap_end - shard.global_index_start
        clip_path = _create_temp_video_clip(shard.path, local_start, local_end)
        try:
            asset = rr.AssetVideo(contents=clip_path.read_bytes(), media_type="video/mp4")
            frame_timestamps_ns = asset.read_frame_timestamps_nanos()
            expected_frames = overlap_end - overlap_start + 1
            if len(frame_timestamps_ns) != expected_frames:
                raise ValueError(
                    f"Video clip frame count mismatch for `{feature_key}`: "
                    f"expected {expected_frames}, got {len(frame_timestamps_ns)}."
                )

            asset_entity = f"{feature_key}/__asset_shard_{shard_index:03d}"
            rec.log(asset_entity, asset, static=True)

            for timestamp_ns, global_index in zip(frame_timestamps_ns, range(overlap_start, overlap_end + 1)):
                rec.reset_time()
                rec.set_time(LEROBOT_TIMELINE_FRAME, sequence=row_lookup[global_index])
                rec.log(
                    feature_key,
                    rr.VideoFrameReference(
                        nanoseconds=int(timestamp_ns),
                        video_reference=asset_entity,
                    ),
                )
        finally:
            clip_path.unlink(missing_ok=True)


def _log_scalar_feature(
    rec: rr.RecordingStream,
    feature_key: str,
    column,
    frame_indices: list[int],
    feature: dict[str, Any],
) -> None:
    rows = _column_to_scalar_rows(column)
    if not rows:
        return

    rec.send_columns(
        feature_key,
        indexes=[rr.TimeColumn(LEROBOT_TIMELINE_FRAME, sequence=frame_indices)],
        columns=rr.Scalars.columns(scalars=rows),
    )

    series_names = _feature_series_names(feature, len(rows[0]))
    if series_names is not None:
        rec.log(feature_key, rr.SeriesLines(names=series_names), static=True)


def _log_task_feature(
    rec: rr.RecordingStream,
    task_index_column,
    frame_indices: list[int],
    episode: EpisodeRecord,
) -> None:
    task_indices = [int(value) if value is not None else -1 for value in task_index_column.to_pylist()]
    for frame_index, task_index in zip(frame_indices, task_indices):
        if task_index < 0:
            continue

        task_label = episode.task_label
        if task_index != episode.task_index:
            task_label = ""
        if not task_label:
            continue

        rec.reset_time()
        rec.set_time(LEROBOT_TIMELINE_FRAME, sequence=frame_index)
        rec.log("task", rr.TextDocument(task_label))


def _column_to_scalar_rows(column) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in column.to_pylist():
        if isinstance(row, list):
            rows.append([float(value) for value in row])
        else:
            rows.append([float(row)])
    return rows


def _feature_series_names(feature: dict[str, Any], width: int) -> list[str] | None:
    names = feature.get("names")
    if not isinstance(names, list):
        return None

    flattened: list[str] = []
    for item in names:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(str(value) for value in item)

    if len(flattened) != width:
        return None
    return flattened


def _short_stream_name(stream_name: str) -> str:
    return stream_name.split(".")[-1]


def _probe_video_stream(video_path: Path) -> dict[str, Any] | None:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None

    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        return None
    return streams[0]


def _create_temp_video_clip(video_path: Path, local_start: int, local_end: int) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise ValueError(f"ffmpeg is required to materialize video clips from `{video_path}`.")

    with tempfile.NamedTemporaryFile(prefix="rerun_lerobot_clip_", suffix=".mp4", delete=False) as handle:
        clip_path = Path(handle.name)
    command = [
        ffmpeg_path,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"trim=start_frame={local_start}:end_frame={local_end + 1},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not clip_path.exists():
        clip_path.unlink(missing_ok=True)
        raise ValueError(
            f"ffmpeg failed to create clip from `{video_path}` "
            f"(frames {local_start}-{local_end}): {result.stderr.strip()}"
        )
    return clip_path
