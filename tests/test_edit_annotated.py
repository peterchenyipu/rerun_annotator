"""Tests for editing annotated RRD files and the three save modes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import rerun as rr
from rerun import recording

from rerun_annotator.schema import (
    ANNOTATION_BOUNDARIES_ENTITY,
    ANNOTATION_SEGMENTS_ENTITY,
    ANNOTATION_SUMMARY_ENTITY,
    SegmentAnnotation,
    build_annotated_rrd_path,
    extract_segments_from_rrd,
    save_annotated_rrd,
    source_has_embedded_segment_annotations,
    strip_annotations_to_rrd,
)

SAMPLE_SEGMENTS = [
    SegmentAnnotation(1, "pick_up", "success", "step", 0.0, 100.0),
    SegmentAnnotation(2, "place_down", "fail", "step", 100.0, 200.0),
]


def _create_source_rrd(path: Path) -> None:
    """Write a minimal source RRD with scalar data on a 'step' timeline."""
    rec = rr.RecordingStream(application_id="test_source")
    rec.save(path)
    for i in range(5):
        rec.set_time("step", sequence=i * 50)
        rec.log("data/value", rr.Scalars([float(i)]))
    rec.flush()


def _create_annotated_rrd(path: Path) -> Path:
    """Create a source RRD, annotate it, and return the annotated path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source.rrd"
        _create_source_rrd(source)
        return save_annotated_rrd(source, SAMPLE_SEGMENTS, output_path=path)


class TestSourceHasEmbeddedSegmentAnnotations(unittest.TestCase):
    def test_unannotated_rrd_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            self.assertFalse(source_has_embedded_segment_annotations(source))

    def test_annotated_rrd_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated = Path(tmpdir) / "annotated.rrd"
            _create_annotated_rrd(annotated)

            self.assertTrue(source_has_embedded_segment_annotations(annotated))


class TestExtractSegmentsFromRrd(unittest.TestCase):
    def test_extracts_correct_number_of_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated = Path(tmpdir) / "annotated.rrd"
            _create_annotated_rrd(annotated)

            segments = extract_segments_from_rrd(annotated)

            self.assertEqual(len(segments), 2)

    def test_extracted_segments_match_originals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated = Path(tmpdir) / "annotated.rrd"
            _create_annotated_rrd(annotated)

            segments = extract_segments_from_rrd(annotated)

            self.assertEqual(segments[0].segment_id, 1)
            self.assertEqual(segments[0].subtask, "pick_up")
            self.assertEqual(segments[0].outcome, "success")
            self.assertEqual(segments[0].timeline, "step")
            self.assertAlmostEqual(segments[0].start_time, 0.0)
            self.assertAlmostEqual(segments[0].end_time, 100.0)

            self.assertEqual(segments[1].segment_id, 2)
            self.assertEqual(segments[1].subtask, "place_down")
            self.assertEqual(segments[1].outcome, "fail")
            self.assertAlmostEqual(segments[1].start_time, 100.0)
            self.assertAlmostEqual(segments[1].end_time, 200.0)

    def test_unannotated_rrd_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            segments = extract_segments_from_rrd(source)

            self.assertEqual(segments, [])


class TestStripAnnotationsToRrd(unittest.TestCase):
    def test_stripped_rrd_has_no_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated = Path(tmpdir) / "annotated.rrd"
            _create_annotated_rrd(annotated)

            stripped = strip_annotations_to_rrd(annotated)

            self.assertFalse(source_has_embedded_segment_annotations(stripped))
            stripped.unlink(missing_ok=True)

    def test_stripped_rrd_preserves_data_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated = Path(tmpdir) / "annotated.rrd"
            _create_annotated_rrd(annotated)

            stripped = strip_annotations_to_rrd(annotated)
            loaded = recording.load_recording(stripped)
            schema_text = str(loaded.schema())

            self.assertIn("/data/value", schema_text)
            stripped.unlink(missing_ok=True)

    def test_stripped_rrd_removes_all_annotation_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated = Path(tmpdir) / "annotated.rrd"
            _create_annotated_rrd(annotated)

            stripped = strip_annotations_to_rrd(annotated)
            loaded = recording.load_recording(stripped)
            entity_paths = {chunk.entity_path for chunk in loaded.chunks()}

            annotation_prefixes = (
                ANNOTATION_SEGMENTS_ENTITY,
                ANNOTATION_BOUNDARIES_ENTITY,
                ANNOTATION_SUMMARY_ENTITY,
            )
            for ep in entity_paths:
                for prefix in annotation_prefixes:
                    self.assertFalse(
                        ep.startswith(prefix),
                        f"Annotation entity {ep} should have been stripped",
                    )
            stripped.unlink(missing_ok=True)


class TestRoundTripEditAnnotatedRrd(unittest.TestCase):
    """Integration test: annotate → extract → strip → re-annotate → verify."""

    def test_full_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Step 1: Create and annotate
            source = tmpdir_path / "source.rrd"
            _create_source_rrd(source)
            annotated = save_annotated_rrd(source, SAMPLE_SEGMENTS, tmpdir_path / "v1.rrd")

            # Step 2: Extract segments from annotated RRD
            extracted = extract_segments_from_rrd(annotated)
            self.assertEqual(len(extracted), 2)

            # Step 3: Strip annotations
            stripped = strip_annotations_to_rrd(annotated)
            self.assertFalse(source_has_embedded_segment_annotations(stripped))

            # Step 4: Modify segments and re-annotate
            modified_segments = [
                SegmentAnnotation(1, "grasp", "success", "step", 0.0, 50.0),
                SegmentAnnotation(2, "lift", "success", "step", 50.0, 150.0),
                SegmentAnnotation(3, "release", "fail", "step", 150.0, 200.0),
            ]
            v2 = save_annotated_rrd(stripped, modified_segments, tmpdir_path / "v2.rrd")

            # Step 5: Verify round-trip
            self.assertTrue(source_has_embedded_segment_annotations(v2))
            final_segments = extract_segments_from_rrd(v2)
            self.assertEqual(len(final_segments), 3)
            self.assertEqual(final_segments[0].subtask, "grasp")
            self.assertEqual(final_segments[1].subtask, "lift")
            self.assertEqual(final_segments[2].subtask, "release")

            stripped.unlink(missing_ok=True)


class TestSaveModes(unittest.TestCase):
    def test_save_overwrite_writes_to_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "episode.rrd"
            _create_source_rrd(source)

            output = save_annotated_rrd(source, SAMPLE_SEGMENTS, output_path=source)

            self.assertEqual(output, source)
            self.assertTrue(source_has_embedded_segment_annotations(output))

    def test_save_duplicate_writes_to_annotated_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "episode.rrd"
            _create_source_rrd(source)

            dup_path = build_annotated_rrd_path(source)
            output = save_annotated_rrd(source, SAMPLE_SEGMENTS, output_path=dup_path)

            self.assertEqual(output, dup_path)
            self.assertEqual(output.name, "episode.annotated.rrd")
            self.assertTrue(source_has_embedded_segment_annotations(output))
            # Original is untouched
            self.assertFalse(source_has_embedded_segment_annotations(source))

    def test_save_as_writes_to_custom_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "episode.rrd"
            _create_source_rrd(source)

            custom = Path(tmpdir) / "subdir" / "custom_output.rrd"
            custom.parent.mkdir(parents=True, exist_ok=True)
            output = save_annotated_rrd(source, SAMPLE_SEGMENTS, output_path=custom)

            self.assertEqual(output, custom)
            self.assertTrue(source_has_embedded_segment_annotations(output))

    def test_save_as_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "episode.rrd"
            _create_source_rrd(source)

            nested = Path(tmpdir) / "a" / "b" / "c" / "out.rrd"
            nested.parent.mkdir(parents=True, exist_ok=True)
            output = save_annotated_rrd(source, SAMPLE_SEGMENTS, output_path=nested)

            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
