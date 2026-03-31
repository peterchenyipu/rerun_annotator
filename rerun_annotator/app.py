from __future__ import annotations

import argparse
import tempfile
from functools import partial
from pathlib import Path
from typing import Sequence

import gradio as gr
from gradio_rerun import Rerun
from gradio_rerun.events import TimelineChange, TimeUpdate

from rerun_annotator.lerobot import (
    DEFAULT_LEROBOT_VIDEO_BACKEND,
    LeRobotDatasetSource,
    LeRobotVideoBackend,
    NativeRrdSource,
    ResolvedSource,
    build_lerobot_manifest_path,
    build_lerobot_output_rrd_path,
    episode_selector_choices,
    materialize_lerobot_episode,
    render_source_summary_markdown,
    resolve_source,
    update_lerobot_annotation_manifest,
)
from rerun_annotator.schema import (
    SegmentAnnotation,
    boundary_table_rows,
    build_segment_annotation,
    build_summary_markdown,
    cleanup_preview_file,
    cleanup_temp_rrd,
    renumber_segments,
    save_annotated_rrd,
    segment_selector_choices,
    segment_table_rows,
    validate_segments,
    write_preview_rrd,
)


def render_cursor_markdown(current_timeline: str, current_time: float) -> str:
    if not current_timeline:
        return "### Viewer Cursor\n_No timeline selected yet._"
    return (
        "### Viewer Cursor\n"
        f"- Timeline: `{current_timeline}`\n"
        f"- Time: `{current_time:.6f}`"
    )


def render_status_markdown(
    source: ResolvedSource | None,
    selected_episode: int | None,
    base_rrd: str | None,
    segments: Sequence[SegmentAnnotation],
    warnings: Sequence[str],
    message: str,
    *,
    manifest_path: str | None = None,
) -> str:
    lines = ["### Status"]
    if isinstance(source, NativeRrdSource):
        lines.append(f"- Source: `{source.path}`")
    elif isinstance(source, LeRobotDatasetSource):
        lines.append(f"- Dataset: `{source.dataset_path}`")
        if selected_episode is not None:
            lines.append(f"- Episode: `{selected_episode}`")
    if base_rrd:
        lines.append(f"- Materialized Base: `{base_rrd}`")
    lines.append(f"- Segments: `{len(segments)}`")
    if manifest_path:
        lines.append(f"- Manifest: `{manifest_path}`")
    if message:
        lines.append(f"- Message: {message}")
    if warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)


def clear_segment_form() -> tuple[str, None, None, str, str]:
    return "", None, None, "", "success"


def prefill_next_segment_form(
    segments: Sequence[SegmentAnnotation],
) -> tuple[str, float | None, None, str, str]:
    if not segments:
        return clear_segment_form()

    last_segment = segments[-1]
    return last_segment.timeline, last_segment.end_time, None, "", "success"


def build_selector_update(segments: Sequence[SegmentAnnotation]):
    return gr.update(
        choices=segment_selector_choices(segments),
        value=None,
    )


def build_episode_update(source: ResolvedSource | None):
    if isinstance(source, LeRobotDatasetSource):
        return gr.update(choices=episode_selector_choices(source), value=None)
    return gr.update(choices=[], value=None)


def build_example_path() -> str:
    example_path = Path("gradio-rerun-viewer/examples/rrt-star.rrd")
    if example_path.exists():
        return str(example_path)
    return ""


def track_current_time(current_timeline: str, evt: TimeUpdate) -> tuple[float, str]:
    return evt.payload.time, render_cursor_markdown(current_timeline, evt.payload.time)


def track_current_timeline_and_time(evt: TimelineChange) -> tuple[str, float, str]:
    return evt.payload.timeline, evt.payload.time, render_cursor_markdown(evt.payload.timeline, evt.payload.time)


def set_start_from_cursor(
    current_timeline: str,
    current_time: float,
    segment_timeline: str,
) -> tuple[str, float]:
    if not current_timeline:
        raise gr.Error("Choose a timeline in the viewer before setting the segment start.")
    if segment_timeline and segment_timeline != current_timeline:
        raise gr.Error("Clear the form before switching the segment to a different timeline.")
    return current_timeline, float(current_time)


def set_end_from_cursor(
    current_timeline: str,
    current_time: float,
    segment_timeline: str,
) -> tuple[str, float]:
    if not current_timeline:
        raise gr.Error("Choose a timeline in the viewer before setting the segment end.")
    if segment_timeline and segment_timeline != current_timeline:
        raise gr.Error("Clear the form before switching the segment to a different timeline.")
    return current_timeline, float(current_time)


def pin_start_to_last_end(
    segments: Sequence[SegmentAnnotation],
) -> tuple[str, float]:
    if not segments:
        raise gr.Error("Add at least one segment before pinning the next start.")

    last_segment = segments[-1]
    return last_segment.timeline, last_segment.end_time


def load_source(
    source_path_input: str,
    previous_preview_rrd: str | None,
    previous_base_rrd: str | None,
):
    if not source_path_input.strip():
        raise gr.Error("Enter a source path before loading.")

    source = resolve_source(Path(source_path_input).expanduser().resolve())
    warnings: list[str] = []
    cleanup_preview_file(previous_preview_rrd)
    cleanup_temp_rrd(previous_base_rrd)

    if isinstance(source, NativeRrdSource):
        preview_rrd = write_preview_rrd(source.path, [], None)
        return (
            str(preview_rrd),
            source,
            None,
            str(source.path),
            str(preview_rrd),
            [],
            "",
            0.0,
            render_cursor_markdown("", 0.0),
            render_source_summary_markdown(source),
            gr.update(visible=False),
            build_episode_update(None),
            segment_table_rows([]),
            boundary_table_rows([]),
            build_summary_markdown([], warnings),
            render_status_markdown(
                source,
                None,
                str(source.path),
                [],
                warnings,
                "Loaded recording. Use the viewer cursor to mark the first segment.",
            ),
            build_selector_update([]),
            *clear_segment_form(),
            gr.update(interactive=False),
            "",
            "",
        )

    return (
        None,
        source,
        None,
        None,
        None,
        [],
        "",
        0.0,
        render_cursor_markdown("", 0.0),
        render_source_summary_markdown(source),
        gr.update(visible=True),
        build_episode_update(source),
        segment_table_rows([]),
        boundary_table_rows([]),
        build_summary_markdown([], warnings),
        render_status_markdown(
            source,
            None,
            None,
            [],
            warnings,
            "Loaded LeRobot dataset. Select an episode and click Load Episode.",
            manifest_path=str(build_lerobot_manifest_path(source.dataset_path)),
        ),
        build_selector_update([]),
        *clear_segment_form(),
        gr.update(interactive=False),
        "",
        str(build_lerobot_manifest_path(source.dataset_path)),
    )


def load_episode(
    source: ResolvedSource | None,
    selected_episode_input: str | None,
    previous_preview_rrd: str | None,
    previous_base_rrd: str | None,
    *,
    video_backend: LeRobotVideoBackend,
):
    if not isinstance(source, LeRobotDatasetSource):
        raise gr.Error("Load a LeRobot dataset before trying to load an episode.")
    if not selected_episode_input:
        raise gr.Error("Choose an episode before loading it.")

    selected_episode = int(selected_episode_input)
    base_rrd = materialize_lerobot_episode(
        source,
        selected_episode,
        previous_base_rrd,
        video_backend=video_backend,
    )
    preview_rrd = write_preview_rrd(base_rrd, [], previous_preview_rrd)
    warnings: list[str] = []

    return (
        str(preview_rrd),
        selected_episode,
        str(base_rrd),
        str(preview_rrd),
        [],
        "",
        0.0,
        render_cursor_markdown("", 0.0),
        segment_table_rows([]),
        boundary_table_rows([]),
        build_summary_markdown([], warnings),
        render_status_markdown(
            source,
            selected_episode,
            str(base_rrd),
            [],
            warnings,
            f"Loaded LeRobot episode #{selected_episode}. Use the viewer cursor to mark the first segment.",
            manifest_path=str(build_lerobot_manifest_path(source.dataset_path)),
        ),
        build_selector_update([]),
        *clear_segment_form(),
        gr.update(interactive=False),
        str(build_lerobot_output_rrd_path(source.dataset_path, selected_episode)),
        str(build_lerobot_manifest_path(source.dataset_path)),
    )


def refresh_annotation_state(
    source: ResolvedSource,
    selected_episode: int | None,
    base_rrd: str,
    previous_preview_rrd: str | None,
    segments: Sequence[SegmentAnnotation],
    message: str,
    *,
    form_values: tuple[str, float | None, float | None, str, str] | None = None,
):
    warnings = validate_segments(segments)
    preview_rrd = write_preview_rrd(Path(base_rrd), segments, previous_preview_rrd)
    next_form_values = clear_segment_form() if form_values is None else form_values
    manifest_path = str(build_lerobot_manifest_path(source.dataset_path)) if isinstance(source, LeRobotDatasetSource) else ""

    return (
        str(preview_rrd),
        str(preview_rrd),
        list(segments),
        segment_table_rows(segments),
        boundary_table_rows(segments),
        build_summary_markdown(segments, warnings),
        render_status_markdown(
            source,
            selected_episode,
            base_rrd,
            segments,
            warnings,
            message,
            manifest_path=manifest_path or None,
        ),
        build_selector_update(segments),
        *next_form_values,
        gr.update(interactive=bool(segments)),
    )


def add_segment(
    source: ResolvedSource | None,
    selected_episode: int | None,
    base_rrd: str | None,
    previous_preview_rrd: str | None,
    segments: list[SegmentAnnotation],
    segment_timeline: str,
    start_time: float | None,
    end_time: float | None,
    subtask: str,
    outcome: str,
):
    if source is None or not base_rrd:
        raise gr.Error("Load a source recording before adding segments.")

    candidate = build_segment_annotation(
        segment_id=len(segments) + 1,
        timeline=segment_timeline,
        start_time=start_time,
        end_time=end_time,
        subtask=subtask,
        outcome=outcome,
    )
    updated_segments = renumber_segments([*segments, candidate])
    return refresh_annotation_state(
        source,
        selected_episode,
        base_rrd,
        previous_preview_rrd,
        updated_segments,
        f"Added segment #{updated_segments[-1].segment_id}.",
        form_values=prefill_next_segment_form(updated_segments),
    )


def load_segment_into_form(
    selected_segment_id: str | None,
    segments: Sequence[SegmentAnnotation],
) -> tuple[str, float, float, str, str]:
    if not selected_segment_id:
        raise gr.Error("Choose a segment to load into the form.")

    selected_id = int(selected_segment_id)
    for segment in segments:
        if segment.segment_id == selected_id:
            return (
                segment.timeline,
                segment.start_time,
                segment.end_time,
                segment.subtask,
                segment.outcome,
            )

    raise gr.Error(f"Segment #{selected_id} no longer exists.")


def update_segment(
    source: ResolvedSource | None,
    selected_episode: int | None,
    base_rrd: str | None,
    previous_preview_rrd: str | None,
    selected_segment_id: str | None,
    segments: list[SegmentAnnotation],
    segment_timeline: str,
    start_time: float | None,
    end_time: float | None,
    subtask: str,
    outcome: str,
):
    if source is None or not base_rrd:
        raise gr.Error("Load a source recording before updating segments.")
    if not selected_segment_id:
        raise gr.Error("Choose a segment to update.")

    selected_id = int(selected_segment_id)
    replacement = build_segment_annotation(
        segment_id=selected_id,
        timeline=segment_timeline,
        start_time=start_time,
        end_time=end_time,
        subtask=subtask,
        outcome=outcome,
    )

    updated_segments: list[SegmentAnnotation] = []
    replaced = False
    for segment in segments:
        if segment.segment_id == selected_id:
            updated_segments.append(replacement)
            replaced = True
        else:
            updated_segments.append(segment)

    if not replaced:
        raise gr.Error(f"Segment #{selected_id} no longer exists.")

    updated_segments = renumber_segments(updated_segments)
    return refresh_annotation_state(
        source,
        selected_episode,
        base_rrd,
        previous_preview_rrd,
        updated_segments,
        f"Updated segment #{selected_id}.",
    )


def delete_segment(
    source: ResolvedSource | None,
    selected_episode: int | None,
    base_rrd: str | None,
    previous_preview_rrd: str | None,
    selected_segment_id: str | None,
    segments: list[SegmentAnnotation],
):
    if source is None or not base_rrd:
        raise gr.Error("Load a source recording before deleting segments.")
    if not selected_segment_id:
        raise gr.Error("Choose a segment to delete.")

    selected_id = int(selected_segment_id)
    remaining_segments = [segment for segment in segments if segment.segment_id != selected_id]
    if len(remaining_segments) == len(segments):
        raise gr.Error(f"Segment #{selected_id} no longer exists.")

    updated_segments = renumber_segments(remaining_segments)
    return refresh_annotation_state(
        source,
        selected_episode,
        base_rrd,
        previous_preview_rrd,
        updated_segments,
        f"Deleted segment #{selected_id}.",
    )


def save_segments(
    source: ResolvedSource | None,
    selected_episode: int | None,
    base_rrd: str | None,
    segments: Sequence[SegmentAnnotation],
) -> tuple[str, str, str]:
    if source is None or not base_rrd:
        raise gr.Error("Load a source recording before saving.")
    if not segments:
        raise gr.Error("Add at least one segment before saving.")

    warnings = validate_segments(segments)
    if isinstance(source, NativeRrdSource):
        output_path = save_annotated_rrd(source.path, segments)
        manifest_path = ""
    else:
        if selected_episode is None:
            raise gr.Error("Load a LeRobot episode before saving annotations.")
        output_path = save_annotated_rrd(
            Path(base_rrd),
            segments,
            output_path=build_lerobot_output_rrd_path(source.dataset_path, selected_episode),
        )
        manifest_path = str(
            update_lerobot_annotation_manifest(source, selected_episode, output_path, list(segments))
        )

    return (
        render_status_markdown(
            source,
            selected_episode,
            base_rrd,
            segments,
            warnings,
            f"Saved annotated recording to `{output_path}`.",
            manifest_path=manifest_path or None,
        ),
        str(output_path),
        manifest_path,
    )


def build_demo(*, video_backend: LeRobotVideoBackend = DEFAULT_LEROBOT_VIDEO_BACKEND) -> gr.Blocks:
    with gr.Blocks(title="Embedded RRD Segment Annotator") as demo:
        source_state = gr.State(None)
        selected_episode_state = gr.State(None)
        base_rrd_state = gr.State(None, delete_callback=cleanup_temp_rrd)
        preview_rrd_state = gr.State(None, delete_callback=cleanup_preview_file)
        segments_state = gr.State([])
        current_timeline_state = gr.State("")
        current_time_state = gr.State(0.0)

        gr.Markdown(
            """
            # Embedded RRD Segment Annotator
            Load a source `.rrd` or a LeRobot dataset directory, scrub the embedded viewer, mark segment boundaries,
            assign a free-text subtask, set `success` or `fail`, and save an annotated `.rrd` with embedded
            annotation entities.
            """
        )

        with gr.Row():
            source_path_input = gr.Textbox(
                label="Source Path",
                value=build_example_path(),
                placeholder="/absolute/path/to/episode.rrd or /absolute/path/to/lerobot_dataset",
            )
            load_button = gr.Button("Load Source", variant="primary")

        with gr.Row():
            with gr.Column(scale=5):
                viewer = Rerun(
                    streaming=False,
                    panel_states={
                        "time": "expanded",
                        "blueprint": "collapsed",
                        "selection": "hidden",
                    },
                    height=820,
                )
            with gr.Column(scale=4):
                source_summary_md = gr.Markdown("### Source Summary\n_No source loaded yet._")
                with gr.Group(visible=False) as episode_group:
                    gr.Markdown("### LeRobot Episode")
                    episode_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Episode",
                        allow_custom_value=False,
                    )
                    load_episode_button = gr.Button("Load Episode")

                cursor_md = gr.Markdown(render_cursor_markdown("", 0.0))
                status_md = gr.Markdown("### Status\n- Segments: `0`")
                summary_md = gr.Markdown(build_summary_markdown([], []))

                segment_table = gr.Dataframe(
                    headers=["segment_id", "subtask", "outcome", "timeline", "start_time", "end_time"],
                    datatype=["number", "str", "str", "str", "number", "number"],
                    value=[],
                    interactive=False,
                    label="Segments",
                )
                boundary_table = gr.Dataframe(
                    headers=["boundary_id", "kind", "segment_id", "timeline", "time", "subtask", "outcome"],
                    datatype=["str", "str", "number", "str", "number", "str", "str"],
                    value=[],
                    interactive=False,
                    label="Boundary Events",
                )

                selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Select Segment",
                    allow_custom_value=False,
                )

                with gr.Group():
                    gr.Markdown("### Segment Form")
                    segment_timeline = gr.Textbox(label="Segment Timeline", interactive=False)
                    with gr.Row():
                        start_time = gr.Number(label="Start Time", precision=6)
                        end_time = gr.Number(label="End Time", precision=6)
                    with gr.Row():
                        set_start_button = gr.Button("Set Start From Cursor")
                        set_end_button = gr.Button("Set End From Cursor")
                    pin_last_end_button = gr.Button("Pin Start To Last End")
                    subtask = gr.Textbox(label="Subtask")
                    outcome = gr.Radio(
                        choices=["success", "fail"],
                        value="success",
                        label="Outcome",
                    )
                    with gr.Row():
                        clear_form_button = gr.Button("Clear Form")
                        load_segment_button = gr.Button("Load Segment")
                    with gr.Row():
                        add_button = gr.Button("Add Segment", variant="primary")
                        update_button = gr.Button("Update Segment")
                        delete_button = gr.Button("Delete Segment", variant="stop")

                with gr.Group():
                    gr.Markdown("### Save")
                    save_button = gr.Button("Save Annotated RRD", interactive=False)
                    save_path = gr.Textbox(label="Annotated Output", interactive=False)
                    manifest_path = gr.Textbox(label="Manifest Output", interactive=False)

        viewer.time_update(
            track_current_time,
            inputs=[current_timeline_state],
            outputs=[current_time_state, cursor_md],
        )
        viewer.timeline_change(
            track_current_timeline_and_time,
            outputs=[current_timeline_state, current_time_state, cursor_md],
        )

        load_button.click(
            load_source,
            inputs=[source_path_input, preview_rrd_state, base_rrd_state],
            outputs=[
                viewer,
                source_state,
                selected_episode_state,
                base_rrd_state,
                preview_rrd_state,
                segments_state,
                current_timeline_state,
                current_time_state,
                cursor_md,
                source_summary_md,
                episode_group,
                episode_selector,
                segment_table,
                boundary_table,
                summary_md,
                status_md,
                selector,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
                save_button,
                save_path,
                manifest_path,
            ],
        )

        load_episode_button.click(
            partial(load_episode, video_backend=video_backend),
            inputs=[source_state, episode_selector, preview_rrd_state, base_rrd_state],
            outputs=[
                viewer,
                selected_episode_state,
                base_rrd_state,
                preview_rrd_state,
                segments_state,
                current_timeline_state,
                current_time_state,
                cursor_md,
                segment_table,
                boundary_table,
                summary_md,
                status_md,
                selector,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
                save_button,
                save_path,
                manifest_path,
            ],
        )

        set_start_button.click(
            set_start_from_cursor,
            inputs=[current_timeline_state, current_time_state, segment_timeline],
            outputs=[segment_timeline, start_time],
        )
        set_end_button.click(
            set_end_from_cursor,
            inputs=[current_timeline_state, current_time_state, segment_timeline],
            outputs=[segment_timeline, end_time],
        )
        pin_last_end_button.click(
            pin_start_to_last_end,
            inputs=[segments_state],
            outputs=[segment_timeline, start_time],
        )
        clear_form_button.click(
            clear_segment_form,
            outputs=[segment_timeline, start_time, end_time, subtask, outcome],
        )
        load_segment_button.click(
            load_segment_into_form,
            inputs=[selector, segments_state],
            outputs=[segment_timeline, start_time, end_time, subtask, outcome],
        )

        add_button.click(
            add_segment,
            inputs=[
                source_state,
                selected_episode_state,
                base_rrd_state,
                preview_rrd_state,
                segments_state,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
            ],
            outputs=[
                viewer,
                preview_rrd_state,
                segments_state,
                segment_table,
                boundary_table,
                summary_md,
                status_md,
                selector,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
                save_button,
            ],
        )
        update_button.click(
            update_segment,
            inputs=[
                source_state,
                selected_episode_state,
                base_rrd_state,
                preview_rrd_state,
                selector,
                segments_state,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
            ],
            outputs=[
                viewer,
                preview_rrd_state,
                segments_state,
                segment_table,
                boundary_table,
                summary_md,
                status_md,
                selector,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
                save_button,
            ],
        )
        delete_button.click(
            delete_segment,
            inputs=[source_state, selected_episode_state, base_rrd_state, preview_rrd_state, selector, segments_state],
            outputs=[
                viewer,
                preview_rrd_state,
                segments_state,
                segment_table,
                boundary_table,
                summary_md,
                status_md,
                selector,
                segment_timeline,
                start_time,
                end_time,
                subtask,
                outcome,
                save_button,
            ],
        )
        save_button.click(
            save_segments,
            inputs=[source_state, selected_episode_state, base_rrd_state, segments_state],
            outputs=[status_md, save_path, manifest_path],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Rerun segment annotator.")
    parser.add_argument(
        "--lerobot-video-backend",
        choices=["asset_video", "video_stream"],
        default=DEFAULT_LEROBOT_VIDEO_BACKEND,
        help=(
            "How LeRobot videos are materialized into the viewer. "
            "`asset_video` is the stable MP4 asset path; `video_stream` tries Rerun VideoStream."
        ),
    )
    args = parser.parse_args()

    demo = build_demo(video_backend=args.lerobot_video_backend)
    allowed_paths = [str(Path.cwd()), tempfile.gettempdir(), "/home/peteop/Desktop"]
    demo.queue().launch(allowed_paths=allowed_paths)


if __name__ == "__main__":
    main()
