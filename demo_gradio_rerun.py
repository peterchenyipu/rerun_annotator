"""
Standalone Gradio demo app for the `gradio_rerun` PyPI component.

This keeps the demo self-contained inside this repository without requiring a
vendored submodule checkout.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
import uuid
from collections import namedtuple
from math import cos, sin

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from fastapi.responses import RedirectResponse
from gradio_rerun import Rerun
from gradio_rerun.events import SelectionChange, TimelineChange, TimeUpdate

ColorGrid = namedtuple("ColorGrid", ["positions", "colors"])
Keypoint = tuple[float, float]
keypoints_per_session_per_sequence_index: dict[str, dict[int, list[Keypoint]]] = {}


def build_color_grid(
    x_count: int = 10,
    y_count: int = 10,
    z_count: int = 10,
    twist: int = 0,
) -> ColorGrid:
    """Create a colored 3D point cloud."""
    grid = np.mgrid[
        slice(-x_count, x_count, x_count * 1j),
        slice(-y_count, y_count, y_count * 1j),
        slice(-z_count, z_count, z_count * 1j),
    ]

    angle = np.linspace(-float(twist) / 2, float(twist) / 2, z_count)
    for z in range(z_count):
        xv, yv, zv = grid[:, :, :, z]
        rot_xv = xv * cos(angle[z]) - yv * sin(angle[z])
        rot_yv = xv * sin(angle[z]) + yv * cos(angle[z])
        grid[:, :, :, z] = [rot_xv, rot_yv, zv]

    positions = np.vstack([xyz.ravel() for xyz in grid])
    colors = np.vstack([
        xyz.ravel()
        for xyz in np.mgrid[
            slice(0, 255, x_count * 1j),
            slice(0, 255, y_count * 1j),
            slice(0, 255, z_count * 1j),
        ]
    ])

    return ColorGrid(positions.T, colors.T.astype(np.uint8))


def get_recording(recording_id: str) -> rr.RecordingStream:
    """Reuse a stable recording ID so new chunks merge in the same viewer."""
    return rr.RecordingStream(application_id="rerun_annotator_demo", recording_id=recording_id)


def streaming_repeated_blur(recording_id: str, img):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()  # type: ignore

    if img is None:
        raise gr.Error("Must provide an image to blur.")

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="image/original"),
            rrb.Spatial2DView(origin="image/blurred"),
        ),
        collapse_panels=True,
    )

    rec.send_blueprint(blueprint)
    rec.set_time("iteration", sequence=0)
    rec.log("image/original", rr.Image(img))
    yield stream.read()

    blur = img
    for i in range(100):
        rec.set_time("iteration", sequence=i)
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        rec.log("image/blurred", rr.Image(blur))
        yield stream.read()


def get_keypoints_for_user_at_sequence_index(request: gr.Request, sequence: int) -> list[Keypoint]:
    if request.session_hash is None:
        raise ValueError("Session hash is None")

    per_sequence = keypoints_per_session_per_sequence_index[request.session_hash]
    if sequence not in per_sequence:
        per_sequence[sequence] = []

    return per_sequence[sequence]


def initialize_instance(request: gr.Request) -> None:
    if request.session_hash is None:
        raise ValueError("Session hash is None")
    keypoints_per_session_per_sequence_index[request.session_hash] = {}


def cleanup_instance(request: gr.Request) -> None:
    if request.session_hash is not None and request.session_hash in keypoints_per_session_per_sequence_index:
        del keypoints_per_session_per_sequence_index[request.session_hash]


def register_keypoint(
    active_recording_id: str,
    current_timeline: str,
    current_time: float,
    request: gr.Request,
    change: SelectionChange,
):
    if active_recording_id == "" or current_timeline != "iteration":
        return

    evt = change.payload
    if len(evt.items) != 1:
        return

    item = evt.items[0]
    if item.type != "entity" or item.position is None:
        return

    rec = get_recording(active_recording_id)
    stream = rec.binary_stream()  # type: ignore
    index = math.floor(current_time)

    keypoints = get_keypoints_for_user_at_sequence_index(request, index)
    keypoints.append((item.position[0], item.position[1]))

    rec.set_time("iteration", sequence=index)
    rec.log(f"{item.entity_path}/keypoint", rr.Points2D(keypoints, radii=2))

    yield stream.read()


def track_current_time(evt: TimeUpdate):
    return evt.payload.time


def track_current_timeline_and_time(evt: TimelineChange):
    return evt.payload.timeline, evt.payload.time


@rr.thread_local_stream("rerun_annotator_demo_cube_rrd")
def create_cube_rrd(x, y, z, pending_cleanup):
    cube = build_color_grid(int(x), int(y), int(z), twist=0)
    rr.log("cube", rr.Points3D(cube.positions, colors=cube.colors, radii=0.5))

    time.sleep(x / 10)

    temp = tempfile.NamedTemporaryFile(prefix="cube_", suffix=".rrd", delete=False)
    pending_cleanup.append(temp.name)

    blueprint = rrb.Spatial3DView(origin="cube")
    rr.save(temp.name, default_blueprint=blueprint)
    return temp.name


def cleanup_cube_rrds(pending_cleanup: list[str]) -> None:
    for file_path in pending_cleanup:
        os.unlink(file_path)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Gradio Rerun Demo
        This app mirrors the upstream `gradio_rerun` demo and shows three common
        integration patterns: live streaming, generated `.rrd` files, and hosted recordings.
        """
    )

    with gr.Tab("Streaming"):
        with gr.Row():
            img = gr.Image(interactive=True, label="Image")
            with gr.Column():
                stream_blur = gr.Button("Stream Repeated Blur")

        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
            )

        recording_id = gr.State(uuid.uuid4())
        current_timeline = gr.State("")
        current_time = gr.State(0.0)

        stream_blur.click(
            streaming_repeated_blur,
            inputs=[recording_id, img],
            outputs=[viewer],
        )
        viewer.selection_change(
            register_keypoint,
            inputs=[recording_id, current_timeline, current_time],
            outputs=[viewer],
        )
        viewer.time_update(track_current_time, outputs=[current_time])
        viewer.timeline_change(track_current_timeline_and_time, outputs=[current_timeline, current_time])

    with gr.Tab("Dynamic RRD"):
        pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_cube_rrds)
        with gr.Row():
            x_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="X Count")
            y_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="Y Count")
            z_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="Z Count")
        with gr.Row():
            create_rrd = gr.Button("Create RRD")
        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
            )
        create_rrd.click(
            create_cube_rrd,
            inputs=[x_count, y_count, z_count, pending_cleanup],
            outputs=[viewer],
        )

    with gr.Tab("Hosted RRD"):
        with gr.Row():
            choose_rrd = gr.Dropdown(
                label="RRD",
                choices=[
                    f"{rr.bindings.get_app_url()}/examples/arkit_scenes.rrd",  # type: ignore[private-use]
                    f"{rr.bindings.get_app_url()}/examples/dna.rrd",  # type: ignore[private-use]
                    f"{rr.bindings.get_app_url()}/examples/plots.rrd",  # type: ignore[private-use]
                ],
            )
        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
            )
        choose_rrd.change(lambda x: x, inputs=[choose_rrd], outputs=[viewer])

    demo.load(initialize_instance)
    demo.close(cleanup_instance)


if __name__ == "__main__":
    app, _, _ = demo.launch(ssr_mode=False, prevent_thread_lock=True)

    @app.get("/custom_component/{component_id}/{environment}/{asset_type}/{file_name}")
    def custom_component_alias(
        component_id: str,
        environment: str,
        asset_type: str,
        file_name: str,
    ):
        return RedirectResponse(
            url=(
                f"/gradio_api/custom_component/{component_id}/{environment}/"
                f"{asset_type}/{file_name}"
            ),
            status_code=307,
        )

    threading.Event().wait()
