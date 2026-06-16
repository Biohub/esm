import io
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
from dna_features_viewer import GraphicFeature, GraphicRecord
from ipywidgets import widgets
from matplotlib import colormaps
from PIL import Image

from esm.sdk.api import FunctionAnnotation
from esm.utils.function.interpro import InterPro, InterProEntryType


@contextmanager
def use_backend(backend):
    original_backend = matplotlib.get_backend()
    matplotlib.use(backend, force=True)
    try:
        yield
    finally:
        matplotlib.use(original_backend, force=True)


def draw_function_annotations(
    annotations: list[FunctionAnnotation], sequence_length: int, interpro_=InterPro()
) -> widgets.Image:
    cmap = colormaps["tab10"]
    colors = [cmap(i) for i in range(len(InterProEntryType))]
    type_colors = dict(zip(InterProEntryType, colors))

    features = []
    for annotation in annotations:
        if annotation.label in interpro_.entries:
            entry = interpro_.entries[annotation.label]
            label = entry.name
            entry_type = entry.type
        else:
            label = annotation.label
            entry_type = InterProEntryType.UNKNOWN

        feature = GraphicFeature(
            start=annotation.start - 1,  # one index -> zero index
            end=annotation.end,
            label=label,
            color=type_colors[entry_type],
            strand=None,
        )
        features.append(feature)

    # Initialize plotting backend
    temp_output = widgets.Output()
    with temp_output:
        fig, ax = plt.subplots()
        temp_output.clear_output()

    buf = io.BytesIO()
    with use_backend("agg"):
        fig, ax = plt.subplots()
        record = GraphicRecord(
            sequence=None, sequence_length=sequence_length, features=features
        )
        record.plot(ax=ax, plot_sequence=False)
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")

    # Load the image from the buffer to get its size
    image = Image.open(buf)
    width, height = image.size
    aspect_ratio = width / height

    # Set the maximum height for the image widget
    max_height = 300
    calculated_width = int(max_height * aspect_ratio)

    buf.seek(0)

    image_widget = widgets.Image(
        value=buf.getvalue(),
        format="png",
        layout=widgets.Layout(width=f"{calculated_width}px", height=f"{max_height}px"),
    )
    buf.close()
    return image_widget
