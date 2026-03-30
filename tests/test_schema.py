from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rerun import recording

from rerun_annotator.schema import (
    ANNOTATION_BOUNDARIES_ENTITY,
    ANNOTATION_SEGMENTS_ENTITY,
    ANNOTATION_SUMMARY_ENTITY,
    SegmentAnnotation,
    build_annotated_rrd_path,
    build_output_rrd_path,
    build_summary_markdown,
    cleanup_temp_rrd,
    materialize_source_recording,
    save_annotated_rrd,
    validate_segments,
)


class SegmentSchemaTests(unittest.TestCase):
    def test_validate_segments_allows_gaps_without_warnings(self) -> None:
        segments = [
            SegmentAnnotation(1, "reach", "success", "step", 0.0, 1.0),
            SegmentAnnotation(2, "place", "fail", "step", 2.0, 3.0),
        ]

        warnings = validate_segments(segments)

        self.assertEqual(warnings, [])

    def test_validate_segments_rejects_overlap(self) -> None:
        segments = [
            SegmentAnnotation(1, "reach", "success", "step", 0.0, 2.0),
            SegmentAnnotation(2, "place", "fail", "step", 1.5, 3.0),
        ]

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_segments(segments)

    def test_validate_segments_rejects_mixed_timelines(self) -> None:
        segments = [
            SegmentAnnotation(1, "reach", "success", "step", 0.0, 1.0),
            SegmentAnnotation(2, "place", "fail", "time", 1.0, 2.0),
        ]

        with self.assertRaisesRegex(ValueError, "same viewer timeline"):
            validate_segments(segments)

    def test_build_summary_markdown_contains_segment_rows(self) -> None:
        segments = [SegmentAnnotation(1, "reach", "success", "step", 0.0, 1.0)]

        summary = build_summary_markdown(segments, [])

        self.assertIn("Trajectory Segments", summary)
        self.assertIn("reach", summary)
        self.assertIn("success", summary)

    def test_build_output_rrd_path_uses_rrd_suffix_for_directory_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "lerobot_episode"
            dataset_dir.mkdir()

            output_path = build_output_rrd_path(dataset_dir)

            self.assertEqual(output_path, dataset_dir.parent / "lerobot_episode.annotated.rrd")

    def test_build_output_rrd_path_uses_rrd_suffix_for_non_rrd_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "episode.mp4"
            video_path.write_bytes(b"demo")

            output_path = build_output_rrd_path(video_path)

            self.assertEqual(output_path, video_path.parent / "episode.annotated.rrd")

    def test_materialize_source_recording_uses_rerun_loader_for_directory_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "lerobot_episode"
            dataset_dir.mkdir()

            fake_stream = mock.Mock()
            with mock.patch("rerun_annotator.schema.rr.RecordingStream", return_value=fake_stream):
                materialized = materialize_source_recording(dataset_dir)

            self.assertTrue(materialized.exists())
            self.assertEqual(fake_stream.log_file_from_path.call_args.args[0], dataset_dir)
            cleanup_temp_rrd(materialized)

    def test_save_annotated_rrd_writes_embedded_columns(self) -> None:
        example_rrd = Path("gradio-rerun-viewer/examples/rrt-star.rrd").resolve()
        self.assertTrue(example_rrd.exists(), "expected example RRD from submodule")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_rrd = tmpdir_path / "episode.rrd"
            shutil.copy2(example_rrd, source_rrd)
            original_bytes = source_rrd.read_bytes()

            output_rrd = save_annotated_rrd(
                source_rrd,
                [
                    SegmentAnnotation(1, "reach", "success", "step", 1.0, 5.0),
                    SegmentAnnotation(2, "place", "fail", "step", 5.0, 8.0),
                ],
            )

            self.assertEqual(output_rrd, build_annotated_rrd_path(source_rrd))
            self.assertTrue(output_rrd.exists())
            self.assertNotEqual(output_rrd, source_rrd)
            self.assertEqual(source_rrd.read_bytes(), original_bytes)

            loaded = recording.load_recording(output_rrd)
            schema = str(loaded.schema())

            self.assertIn(f"{ANNOTATION_SEGMENTS_ENTITY}:segment_id", schema)
            self.assertIn(f"{ANNOTATION_SEGMENTS_ENTITY}:subtask", schema)
            self.assertIn(f"{ANNOTATION_SEGMENTS_ENTITY}:outcome", schema)
            self.assertIn(f"{ANNOTATION_SEGMENTS_ENTITY}:timeline", schema)
            self.assertIn(f"{ANNOTATION_SEGMENTS_ENTITY}:start_time", schema)
            self.assertIn(f"{ANNOTATION_SEGMENTS_ENTITY}:end_time", schema)
            self.assertIn(f"{ANNOTATION_SUMMARY_ENTITY}:TextDocument:text", schema)
            self.assertIn(f"{ANNOTATION_BOUNDARIES_ENTITY}:TextLog:text", schema)

    def test_save_annotated_rrd_keeps_timestamp_timeline_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_rrd = tmpdir_path / "timestamp_episode.rrd"

            self._write_timestamp_source_rrd(source_rrd)
            output_rrd = save_annotated_rrd(
                source_rrd,
                [SegmentAnnotation(1, "reach", "success", "episode_time", 1_000_000_000.0, 2_000_000_000.0)],
            )

            schema = recording.load_recording(output_rrd).schema()
            index_names = [column.name for column in schema.index_columns()]

            self.assertIn("episode_time", index_names)
            self.assertIn("annotation_row", index_names)
            self.assertEqual(index_names.count("episode_time"), 1)

    @staticmethod
    def _write_timestamp_source_rrd(path: Path) -> None:
        from datetime import datetime, timezone

        import rerun as rr

        rec = rr.RecordingStream(application_id="timestamp_source")
        rec.save(path)
        rec.set_time("episode_time", timestamp=datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc))
        rec.log("trajectory/value", rr.Scalars([1.0]))
        rec.set_time("episode_time", timestamp=datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
        rec.log("trajectory/value", rr.Scalars([2.0]))
        rec.flush()


if __name__ == "__main__":
    unittest.main()
