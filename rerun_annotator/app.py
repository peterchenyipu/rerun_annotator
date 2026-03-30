from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import gradio as gr
from gradio_rerun import Rerun
from gradio_rerun.events import TimelineChange, TimeUpdate

from rerun_annotator.schema import (
    SegmentAnnotation,
    boundary_table_rows,
    build_segment_annotation,
    build_summary_markdown,
    cleanup_preview_file,
    ensure_source_can_be_annotated,
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
    source_rrd: str | None,
    segments: Sequence[SegmentAnnotation],
    warnings: Sequence[str],
    message: str,
) -> str:
    lines = ["### Status"]
    if source_rrd:
        lines.append(f"- Recording: `{source_rrd}`")
    lines.append(f"- Segments: `{len(segments)}`")
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


def load_rrd(
    source_rrd_input: str,
    previous_preview_rrd: str | None,
) -> tuple[
    str,
    str,
    str,
    list[SegmentAnnotation],
    str,
    float,
    str,
    list[list[str | int | float]],
    list[list[str | int | float]],
    str,
    str,
    object,
    str,
    None,
    None,
    str,
    str,
    gr.Button,
    str,
]:
    source_rrd = Path(source_rrd_input).expanduser().resolve()
    ensure_source_can_be_annotated(source_rrd)

    preview_rrd = write_preview_rrd(source_rrd, [], previous_preview_rrd)
    warnings: list[str] = []

    return (
        str(preview_rrd),
        str(source_rrd),
        str(preview_rrd),
        [],
        "",
        0.0,
        render_cursor_markdown("", 0.0),
        segment_table_rows([]),
        boundary_table_rows([]),
        build_summary_markdown([], warnings),
        render_status_markdown(
            str(source_rrd),
            [],
            warnings,
            "Loaded recording. Use the viewer cursor to mark the first segment.",
        ),
        build_selector_update([]),
        *clear_segment_form(),
        gr.update(interactive=False),
        "",
    )


def refresh_annotation_state(
    source_rrd: str,
    previous_preview_rrd: str | None,
    segments: Sequence[SegmentAnnotation],
    message: str,
    *,
    form_values: tuple[str, float | None, float | None, str, str] | None = None,
) -> tuple[
    str,
    str,
    list[SegmentAnnotation],
    list[list[str | int | float]],
    list[list[str | int | float]],
    str,
    str,
    object,
    str,
    None,
    None,
    str,
    str,
    gr.Button,
]:
    warnings = validate_segments(segments)
    preview_rrd = write_preview_rrd(Path(source_rrd), segments, previous_preview_rrd)
    normalized_segments = list(segments)
    next_form_values = clear_segment_form() if form_values is None else form_values

    return (
        str(preview_rrd),
        str(preview_rrd),
        normalized_segments,
        segment_table_rows(normalized_segments),
        boundary_table_rows(normalized_segments),
        build_summary_markdown(normalized_segments, warnings),
        render_status_markdown(source_rrd, normalized_segments, warnings, message),
        build_selector_update(normalized_segments),
        *next_form_values,
        gr.update(interactive=bool(normalized_segments)),
    )


def add_segment(
    source_rrd: str | None,
    previous_preview_rrd: str | None,
    segments: list[SegmentAnnotation],
    segment_timeline: str,
    start_time: float | None,
    end_time: float | None,
    subtask: str,
    outcome: str,
) -> tuple[
    str,
    str,
    list[SegmentAnnotation],
    list[list[str | int | float]],
    list[list[str | int | float]],
    str,
    str,
    object,
    str,
    None,
    None,
    str,
    str,
    gr.Button,
]:
    if not source_rrd:
        raise gr.Error("Load an RRD file before adding segments.")

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
        source_rrd,
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
    source_rrd: str | None,
    previous_preview_rrd: str | None,
    selected_segment_id: str | None,
    segments: list[SegmentAnnotation],
    segment_timeline: str,
    start_time: float | None,
    end_time: float | None,
    subtask: str,
    outcome: str,
) -> tuple[
    str,
    str,
    list[SegmentAnnotation],
    list[list[str | int | float]],
    list[list[str | int | float]],
    str,
    str,
    object,
    str,
    None,
    None,
    str,
    str,
    gr.Button,
]:
    if not source_rrd:
        raise gr.Error("Load an RRD file before updating segments.")
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
        source_rrd,
        previous_preview_rrd,
        updated_segments,
        f"Updated segment #{selected_id}.",
    )


def delete_segment(
    source_rrd: str | None,
    previous_preview_rrd: str | None,
    selected_segment_id: str | None,
    segments: list[SegmentAnnotation],
) -> tuple[
    str,
    str,
    list[SegmentAnnotation],
    list[list[str | int | float]],
    list[list[str | int | float]],
    str,
    str,
    object,
    str,
    None,
    None,
    str,
    str,
    gr.Button,
]:
    if not source_rrd:
        raise gr.Error("Load an RRD file before deleting segments.")
    if not selected_segment_id:
        raise gr.Error("Choose a segment to delete.")

    selected_id = int(selected_segment_id)
    remaining_segments = [segment for segment in segments if segment.segment_id != selected_id]
    if len(remaining_segments) == len(segments):
        raise gr.Error(f"Segment #{selected_id} no longer exists.")

    updated_segments = renumber_segments(remaining_segments)
    return refresh_annotation_state(
        source_rrd,
        previous_preview_rrd,
        updated_segments,
        f"Deleted segment #{selected_id}.",
    )


def save_segments(source_rrd: str | None, segments: Sequence[SegmentAnnotation]) -> tuple[str, str]:
    if not source_rrd:
        raise gr.Error("Load an RRD file before saving.")
    if not segments:
        raise gr.Error("Add at least one segment before saving.")

    output_path = save_annotated_rrd(Path(source_rrd), segments)
    warnings = validate_segments(segments)
    return (
        render_status_markdown(
            source_rrd,
            segments,
            warnings,
            f"Saved annotated recording to `{output_path}`.",
        ),
        str(output_path),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Embedded RRD Segment Annotator") as demo:
        source_rrd_state = gr.State(None)
        preview_rrd_state = gr.State(None, delete_callback=cleanup_preview_file)
        segments_state = gr.State([])
        current_timeline_state = gr.State("")
        current_time_state = gr.State(0.0)

        gr.Markdown(
            """
            # Embedded RRD Segment Annotator
            Load one `.rrd`, scrub the embedded viewer, mark segment boundaries, assign a free-text subtask, set
            `success` or `fail`, and save a sibling annotated recording with embedded annotation entities.
            """
        )

        with gr.Row():
            source_rrd_input = gr.Textbox(
                label="Source RRD Path",
                value=build_example_path(),
                placeholder="/absolute/path/to/episode.rrd",
            )
            load_button = gr.Button("Load RRD", variant="primary")

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
            load_rrd,
            inputs=[source_rrd_input, preview_rrd_state],
            outputs=[
                viewer,
                source_rrd_state,
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
                source_rrd_state,
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
                source_rrd_state,
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
            inputs=[source_rrd_state, preview_rrd_state, selector, segments_state],
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
            inputs=[source_rrd_state, segments_state],
            outputs=[status_md, save_path],
        )

    return demo


def main() -> None:
    demo = build_demo()
    allowed_paths = [str(Path.cwd()), tempfile.gettempdir()]
    demo.queue().launch(allowed_paths=allowed_paths)


if __name__ == "__main__":
    main()
