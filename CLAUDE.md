# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`rerun-annotator` is an annotation tool for [Rerun](https://rerun.io/) built on top of [Gradio](https://gradio.app/). It uses the published `gradio-rerun` package as its core dependency, which embeds the Rerun viewer as a custom Gradio component.

## Setup & Running

This project uses `uv` for dependency management (Python 3.10 required).

```bash
# Install dependencies
uv sync

# Run the main app
uv run python main.py

# Run with a custom blueprint
uv run python main.py --blueprint my_layout.rbl

# Disable blueprint (use default auto-generated layout)
uv run python main.py --blueprint ""
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lerobot-video-backend` | `asset_video` | How LeRobot videos are materialized (`asset_video` or `video_stream`) |
| `--blueprint` | `bp.rbl` | Path to a Rerun `.rbl` blueprint file. Empty string disables. |

## Architecture

### Dependency relationship

`pyproject.toml` depends directly on `gradio-rerun` from PyPI. The repository no longer vendors the component source as a git submodule.

### Local demo entrypoints

- **`demo_gradio_rerun.py`** — Standalone demo showing streaming blur, dynamic RRD loading, hosted RRD loading, and keypoint annotation flows using the installed `gradio_rerun` package.
- **`scripts/run_root_gradio_rerun_demo.sh`** — Launches the demo from the repository root after `uv sync`.
- **`scripts/run_gradio_rerun_demo.sh`** — Compatibility wrapper that forwards to the root demo runner.

### Key patterns for building with `gradio_rerun`

**Streaming data** (preferred): Use `rr.binary_stream()` and `yield stream.read()` in a generator function wired to a Gradio event. The `Rerun` component must be initialized with `streaming=True`.

**File-based data**: Return a file path (`.rrd`) or URL string from a callback. Gradio serves local files; remote URLs are passed directly to the viewer.

**Session state**: Use `gr.State(uuid.uuid4())` for per-session recording IDs to avoid mixing data across users. Use `gr.Request` in callbacks to access `request.session_hash` for per-user server-side state.

**Events from viewer**: Wire `viewer.selection_change`, `viewer.time_update`, `viewer.timeline_change` to callbacks. Event payloads are typed objects accessed via `evt.payload`.

**Panel control**: Pass `panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}` to the `Rerun(...)` constructor.

**Custom blueprints**: The annotator supports loading external `.rbl` blueprint files via `--blueprint`. The blueprint is re-stamped to match the recording's `application_id` using `rerun rrd route`, then merged into the preview RRD with `rerun rrd merge`. When a custom blueprint is provided, the default programmatic blueprint (`build_annotation_blueprint()` in `schema.py`) is skipped. Saved (final) RRDs always embed the default blueprint for portability.

**Editing annotated RRDs**: Loading an already-annotated `.rrd` file extracts existing segments (`extract_segments_from_rrd` in `schema.py`), strips annotation entities from the source (`strip_annotations_to_rrd`), and pre-populates the segment table for editing. Three save modes are available: **Save (Overwrite)** writes back to the original source path, **Save (Duplicate)** writes to `<source>.annotated.rrd`, and **Save As** writes to a user-specified path.
