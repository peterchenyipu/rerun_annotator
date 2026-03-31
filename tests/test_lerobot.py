from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rerun import recording

from rerun_annotator.lerobot import (
    LeRobotDatasetSource,
    build_lerobot_manifest_path,
    build_lerobot_output_rrd_path,
    materialize_lerobot_episode,
    resolve_source,
    update_lerobot_annotation_manifest,
)
from rerun_annotator.schema import SegmentAnnotation, cleanup_temp_rrd


class LeRobotSourceTests(unittest.TestCase):
    def test_resolve_source_detects_lerobot_dataset_and_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = self._create_fake_lerobot_dataset(Path(tmpdir))

            source = resolve_source(dataset_path)

            self.assertIsInstance(source, LeRobotDatasetSource)
            self.assertEqual(source.dataset_path, dataset_path)
            self.assertEqual(len(source.episodes), 2)
            self.assertEqual(source.video_streams, ["observation.images.top_image"])
            self.assertEqual(source.episodes[0].row_count, 3)
            self.assertEqual(source.episodes[1].global_index_start, 3)

    def test_materialize_lerobot_episode_creates_rrd_with_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = self._create_fake_lerobot_dataset(Path(tmpdir))
            source = resolve_source(dataset_path)
            self.assertIsInstance(source, LeRobotDatasetSource)

            materialized = materialize_lerobot_episode(source, 1)
            try:
                self.assertTrue(materialized.exists())
                loaded = recording.load_recording(materialized)
                schema = str(loaded.schema())
                index_names = [column.name for column in loaded.schema().index_columns()]

                self.assertIn("frame_index", index_names)
                self.assertNotIn("episode_time", index_names)
                self.assertIn("/observation.images.top_image", schema)
                self.assertIn("/observation.state:Scalars:scalars", schema)
                self.assertIn("/action:Scalars:scalars", schema)
                self.assertIn("/task:TextDocument:text", schema)
            finally:
                cleanup_temp_rrd(materialized)

    def test_materialize_lerobot_episode_with_video_stream_backend_creates_rrd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = self._create_fake_lerobot_dataset(Path(tmpdir))
            source = resolve_source(dataset_path)
            self.assertIsInstance(source, LeRobotDatasetSource)

            materialized = materialize_lerobot_episode(source, 1, video_backend="video_stream")
            try:
                self.assertTrue(materialized.exists())
                loaded = recording.load_recording(materialized)
                schema = str(loaded.schema())

                self.assertIn("/observation.images.top_image:VideoStream:sample", schema)
            finally:
                cleanup_temp_rrd(materialized)

    def test_update_lerobot_annotation_manifest_replaces_existing_episode_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = self._create_fake_lerobot_dataset(Path(tmpdir))
            source = resolve_source(dataset_path)
            self.assertIsInstance(source, LeRobotDatasetSource)

            output_path = build_lerobot_output_rrd_path(dataset_path, 1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"rrd")

            first_manifest = update_lerobot_annotation_manifest(
                source,
                1,
                output_path,
                [SegmentAnnotation(1, "reach", "success", "episode_time", 0.0, 1.0)],
            )
            second_manifest = update_lerobot_annotation_manifest(
                source,
                1,
                output_path,
                [
                    SegmentAnnotation(1, "reach", "success", "episode_time", 0.0, 1.0),
                    SegmentAnnotation(2, "place", "fail", "episode_time", 1.0, 2.0),
                ],
            )

            self.assertEqual(first_manifest, build_lerobot_manifest_path(dataset_path))
            self.assertEqual(second_manifest, first_manifest)

            manifest_table = pq.read_table(second_manifest)
            self.assertEqual(manifest_table.num_rows, 1)
            self.assertEqual(manifest_table["episode_index"][0].as_py(), 1)
            self.assertEqual(manifest_table["segment_count"][0].as_py(), 2)
            self.assertEqual(manifest_table["fail_count"][0].as_py(), 1)

    @staticmethod
    def _create_fake_lerobot_dataset(root: Path) -> Path:
        dataset_path = root / "mini_lerobot"
        (dataset_path / "meta").mkdir(parents=True)
        (dataset_path / "data" / "chunk-000").mkdir(parents=True)
        (dataset_path / "videos" / "observation.images.top_image" / "chunk-000").mkdir(parents=True)

        info = {
            "codebase_version": "v3.0",
            "robot_type": "testbot",
            "total_episodes": 2,
            "total_frames": 5,
            "total_tasks": 1,
            "fps": 5,
            "features": {
                "observation.images.top_image": {
                    "dtype": "video",
                    "shape": [16, 16, 3],
                    "info": {"video.fps": 5},
                },
                "observation.state": {"dtype": "float32", "shape": [2], "names": ["state"]},
                "action": {"dtype": "float32", "shape": [2], "names": ["action"]},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        (dataset_path / "meta" / "info.json").write_text(json.dumps(info))

        task_table = pa.Table.from_pydict(
            {
                "task_index": [0],
                "__index_level_0__": ["pick and place"],
            }
        )
        pq.write_table(task_table, dataset_path / "meta" / "tasks.parquet")

        data_table = pa.Table.from_arrays(
            [
                pa.array(
                    [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [1.0, 1.1], [1.1, 1.2]],
                    type=pa.list_(pa.float32(), 2),
                ),
                pa.array(
                    [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [1.5, 1.6], [1.6, 1.7]],
                    type=pa.list_(pa.float32(), 2),
                ),
                pa.array([0.0, 0.2, 0.4, 0.0, 0.2], type=pa.float32()),
                pa.array([0, 1, 2, 0, 1], type=pa.int64()),
                pa.array([0, 0, 0, 1, 1], type=pa.int64()),
                pa.array([0, 1, 2, 3, 4], type=pa.int64()),
                pa.array([0, 0, 0, 0, 0], type=pa.int64()),
            ],
            names=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )
        pq.write_table(data_table, dataset_path / "data" / "chunk-000" / "file-000.parquet")

        video_path = dataset_path / "videos" / "observation.images.top_image" / "chunk-000" / "file-000.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
        if not writer.isOpened():
            raise RuntimeError("Failed to create test MP4.")
        for idx in range(5):
            frame = np.full((16, 16, 3), idx * 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        return dataset_path


if __name__ == "__main__":
    unittest.main()
