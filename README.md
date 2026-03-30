# Rerun Annotator

Launch the embedded trajectory segment annotator with:

```bash
uv run -m rerun_annotator
```

The app loads one source `.rrd`, lets you create ordered non-overlapping segments with a free-text
subtask and `success` / `fail` outcome, previews the embedded annotation entities in the Rerun
viewer, and saves a sibling output like `episode.annotated.rrd`.
