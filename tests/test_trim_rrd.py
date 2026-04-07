"""Tests for the RRD trim feature."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import rerun as rr
from rerun import recording

from rerun_annotator.schema import (
    SegmentAnnotation,
    TrimRegion,
    filter_segments_after_trim,
    trim_region_table_rows,
    trim_rrd,
)


def _create_source_rrd(path: Path, num_steps: int = 10) -> None:
    """Write a minimal source RRD with scalar data on a 'step' timeline."""
    rec = rr.RecordingStream(application_id="test_trim")
    rec.save(path)
    for i in range(num_steps):
        rec.set_time("step", sequence=i * 10)
        rec.log("data/value", rr.Scalars([float(i)]))
    rec.flush()


def _create_multi_stream_rrd(path: Path) -> None:
    """Write an RRD with multiple entity streams on the same timeline."""
    rec = rr.RecordingStream(application_id="test_trim_multi")
    rec.save(path)
    for i in range(10):
        rec.set_time("step", sequence=i * 10)
        rec.log("sensor/a", rr.Scalars([float(i)]))
        rec.log("sensor/b", rr.Scalars([float(i * 2)]))
    rec.flush()


def _get_step_values(rrd_path: Path) -> dict[str, list[int]]:
    """Load an RRD and return {entity_path: sorted list of step values}."""
    r = recording.load_recording(rrd_path)
    result: dict[str, list[int]] = {}
    for chunk in r.chunks():
        rb = chunk.to_record_batch()
        if "step" not in rb.schema.names:
            continue
        ep = chunk.entity_path
        values = [int(v) for v in rb.column("step").to_pylist() if v is not None]
        result.setdefault(ep, []).extend(values)
    for ep in result:
        result[ep] = sorted(set(result[ep]))
    return result


# ---------------------------------------------------------------------------
# filter_segments_after_trim
# ---------------------------------------------------------------------------


class TestFilterSegmentsAfterTrim(unittest.TestCase):
    SEGMENTS = [
        SegmentAnnotation(1, "A", "success", "step", 0.0, 30.0),
        SegmentAnnotation(2, "B", "success", "step", 30.0, 60.0),
        SegmentAnnotation(3, "C", "fail", "step", 60.0, 90.0),
    ]

    def test_keep_mode_keeps_inside(self) -> None:
        kept = filter_segments_after_trim(
            self.SEGMENTS, [TrimRegion(0.0, 60.0)], "keep",
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].subtask, "A")
        self.assertEqual(kept[1].subtask, "B")

    def test_keep_mode_drops_outside(self) -> None:
        kept = filter_segments_after_trim(
            self.SEGMENTS, [TrimRegion(30.0, 60.0)], "keep",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].subtask, "B")

    def test_remove_mode_drops_overlapping(self) -> None:
        kept = filter_segments_after_trim(
            self.SEGMENTS, [TrimRegion(25.0, 65.0)], "remove",
        )
        # Segments A (0-30) overlaps [25,65], B (30-60) is inside, C (60-90) overlaps
        self.assertEqual(len(kept), 0)

    def test_remove_mode_keeps_non_overlapping(self) -> None:
        kept = filter_segments_after_trim(
            self.SEGMENTS, [TrimRegion(35.0, 55.0)], "remove",
        )
        # Only B (30-60) overlaps [35,55]
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].subtask, "A")
        self.assertEqual(kept[1].subtask, "C")

    def test_segments_renumbered(self) -> None:
        kept = filter_segments_after_trim(
            self.SEGMENTS, [TrimRegion(60.0, 90.0)], "keep",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].segment_id, 1)  # renumbered from 3


# ---------------------------------------------------------------------------
# trim_region_table_rows
# ---------------------------------------------------------------------------


class TestTrimRegionTableRows(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(trim_region_table_rows([]), [])

    def test_single_region(self) -> None:
        rows = trim_region_table_rows([TrimRegion(10.0, 50.0)])
        self.assertEqual(rows, [[1, 10.0, 50.0]])

    def test_multiple_regions(self) -> None:
        regions = [TrimRegion(0.0, 20.0), TrimRegion(40.0, 80.0)]
        rows = trim_region_table_rows(regions)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[1][0], 2)


# ---------------------------------------------------------------------------
# trim_rrd  (uses rerun CLI split + merge)
# ---------------------------------------------------------------------------


class TestTrimRrd(unittest.TestCase):
    def test_keep_middle_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            trimmed = trim_rrd(source, "step", [TrimRegion(20.0, 60.0)], "keep")

            self.assertTrue(trimmed.exists())
            steps = _get_step_values(trimmed)
            vals = steps.get("/data/value", [])
            # Region [20, 60] inclusive — steps 20, 30, 40, 50, 60 should be kept
            self.assertTrue(all(20 <= v <= 60 for v in vals))
            self.assertIn(20, vals)
            self.assertIn(60, vals)
            trimmed.unlink(missing_ok=True)

    def test_remove_middle_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            trimmed = trim_rrd(source, "step", [TrimRegion(30.0, 60.0)], "remove")

            self.assertTrue(trimmed.exists())
            steps = _get_step_values(trimmed)
            vals = steps.get("/data/value", [])
            # Steps 30, 40, 50, 60 should NOT be present (inclusive end)
            for v in vals:
                self.assertFalse(30 <= v <= 60, f"Step {v} should have been removed")
            trimmed.unlink(missing_ok=True)

    def test_empty_regions_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            with self.assertRaises(ValueError):
                trim_rrd(source, "step", [], "keep")

    def test_keep_all_returns_valid_rrd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            trimmed = trim_rrd(source, "step", [TrimRegion(0.0, 90.0)], "keep")

            self.assertTrue(trimmed.exists())
            self.assertGreater(trimmed.stat().st_size, 0)
            trimmed.unlink(missing_ok=True)

    def test_multiple_keep_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)

            trimmed = trim_rrd(
                source, "step",
                [TrimRegion(0.0, 20.0), TrimRegion(60.0, 90.0)],
                "keep",
            )

            self.assertTrue(trimmed.exists())
            steps = _get_step_values(trimmed)
            vals = steps.get("/data/value", [])
            # Should have 0, 10, 20 and 60, 70, 80, 90 — nothing in (20, 60)
            for v in vals:
                self.assertFalse(20 < v < 60, f"Step {v} should not be present")
            self.assertIn(20, vals)
            self.assertIn(60, vals)
            trimmed.unlink(missing_ok=True)

    def test_remove_last_data_point(self) -> None:
        """Removing a region that includes the last timestamp should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)  # steps 0, 10, 20, ..., 90

            # Remove the last two steps [80, 90] — both should be gone
            trimmed = trim_rrd(source, "step", [TrimRegion(80.0, 90.0)], "remove")

            self.assertTrue(trimmed.exists())
            steps = _get_step_values(trimmed)
            vals = steps.get("/data/value", [])
            self.assertNotIn(80, vals)
            self.assertNotIn(90, vals)
            self.assertIn(70, vals)  # 70 should still be there
            trimmed.unlink(missing_ok=True)

    def test_keep_up_to_last_data_point(self) -> None:
        """Keeping a region that includes the last timestamp should include it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_source_rrd(source)  # steps 0, 10, 20, ..., 90

            trimmed = trim_rrd(source, "step", [TrimRegion(60.0, 90.0)], "keep")

            self.assertTrue(trimmed.exists())
            steps = _get_step_values(trimmed)
            vals = steps.get("/data/value", [])
            self.assertIn(90, vals)  # last step must be included
            self.assertIn(60, vals)
            self.assertNotIn(50, vals)
            trimmed.unlink(missing_ok=True)

    def test_multi_stream_trimmed_uniformly(self) -> None:
        """All entity streams should end up with the same step values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.rrd"
            _create_multi_stream_rrd(source)

            trimmed = trim_rrd(source, "step", [TrimRegion(20.0, 60.0)], "keep")

            steps = _get_step_values(trimmed)
            self.assertIn("/sensor/a", steps)
            self.assertIn("/sensor/b", steps)
            self.assertEqual(steps["/sensor/a"], steps["/sensor/b"],
                             "Both streams should have identical step values after trim")
            trimmed.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
