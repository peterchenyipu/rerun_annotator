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
```

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
