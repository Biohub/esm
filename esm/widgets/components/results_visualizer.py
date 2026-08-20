from datetime import datetime
from functools import partial
from typing import Any, Callable, Literal

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from esm.sdk.api import ESMProtein
from esm.widgets.utils.drawing.draw_category_array import draw_data_array
from esm.widgets.utils.drawing.draw_function_annotations import (
    draw_function_annotations,
)
from esm.widgets.utils.drawing.draw_protein_structure import draw_protein_structure
from esm.widgets.utils.serialization import (
    create_download_button_from_buffer,
    protein_to_pdb_buffer,
)


def create_results_visualizer(
    modality: str,
    samples: list[ESMProtein],
    items_per_page: int = 4,
    copy_to_prompt_callback: Callable[
        [
            Literal[
                "sequence", "coordinates", "secondary_structure", "sasa", "function"
            ],
            Any,
        ],
        None,
    ]
    | None = None,
    include_title: bool = True,
) -> widgets.Widget:
    if modality == "structure":
        # Sort structures by pTM
        samples = sorted(
            samples,
            key=lambda item: item.ptm.item() if item.ptm is not None else 0,
            reverse=True,
        )

    total_pages = (len(samples) + items_per_page - 1) // items_per_page
    current_page = 1

    output = widgets.VBox()
    page_label = widgets.Label(value=f"Page {current_page} of {total_pages}")
    next_button = widgets.Button(
        description="Next", disabled=current_page == total_pages
    )
    prev_button = widgets.Button(description="Previous", disabled=current_page == 1)

    def update_page():
        nonlocal current_page
        start = (current_page - 1) * items_per_page
        end = start + items_per_page
        items = samples[start:end]

        if modality == "sequence":
            results_page = create_sequence_results_page(
                items,
                copy_to_prompt_callback=partial(copy_to_prompt_callback, "sequence")
                if copy_to_prompt_callback
                else None,
            )
        elif modality == "sasa":
            results_page = create_sasa_results_page(
                items,
                copy_to_prompt_callback=partial(copy_to_prompt_callback, "sasa")
                if copy_to_prompt_callback
                else None,
            )
        elif modality == "secondary_structure":
            results_page = create_secondary_structure_results_page(
                items,
                copy_to_prompt_callback=partial(
                    copy_to_prompt_callback, "secondary_structure"
                )
                if copy_to_prompt_callback
                else None,
            )
        elif modality == "structure":
            results_page = create_structure_results_page(
                items,
                copy_to_prompt_callback=partial(copy_to_prompt_callback, "coordinates")
                if copy_to_prompt_callback
                else None,
            )
        elif modality == "function":
            results_page = create_function_annotations_results_page(
                items,
                copy_to_prompt_callback=partial(copy_to_prompt_callback, "function")
                if copy_to_prompt_callback
                else None,
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")

        output.children = [results_page]
        page_label.value = f"Page {current_page} of {total_pages}"
        prev_button.disabled = current_page == 1
        next_button.disabled = current_page == total_pages

    def on_next_button_clicked(b):
        nonlocal current_page
        if current_page < total_pages:
            current_page += 1
            update_page()

    def on_prev_button_clicked(b):
        nonlocal current_page
        if current_page > 1:
            current_page -= 1
            update_page()

    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)

    update_page()

    results_ui = widgets.VBox([])
    title = (widgets.HTML(value="<h1>Generated Samples</h1>"),)
    nav_bar = widgets.HBox([prev_button, next_button, page_label])

    if include_title:
        results_ui.children += title
    if total_pages > 1:
        results_ui.children += (nav_bar,)
    results_ui.children += (output,)
    return results_ui


def add_line_breaks(sequence: str, line_length: int = 120) -> str:
    return "<br>".join(
        [sequence[i : i + line_length] for i in range(0, len(sequence), line_length)]
    )


def create_sequence_results_page(
    items: list[ESMProtein],
    line_length: int = 120,
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    sequence_items = []
    for item in items:
        copy_to_prompt_button = widgets.Button(
            description="Copy to Prompt", disabled=copy_to_prompt_callback is None
        )
        if copy_to_prompt_callback:
            copy_to_prompt_button.on_click(
                lambda b: copy_to_prompt_callback(item.sequence)
            )

        if copy_to_prompt_callback:
            entry = widgets.VBox(
                [
                    copy_to_prompt_button,
                    widgets.HTML(
                        value=f'<pre style="white-space: pre-wrap; font-family: monospace;">{add_line_breaks(item.sequence, line_length) if item.sequence else "No sequence"}</pre>'
                    ),
                ],
                layout={"border": "1px solid gray"},
            )
        else:
            entry = widgets.VBox(
                [
                    widgets.HTML(
                        value=f'<pre style="white-space: pre-wrap; font-family: monospace;">{add_line_breaks(item.sequence, line_length) if item.sequence else "No sequence"}</pre>'
                    )
                ],
                layout={"border": "1px solid gray"},
            )

        sequence_items.append(entry)

    return widgets.VBox(sequence_items)


def create_sasa_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    sasa_items = []
    for item in items:
        copy_to_prompt_button = widgets.Button(
            description="Copy to Prompt", disabled=copy_to_prompt_callback is None
        )
        if copy_to_prompt_callback:
            copy_to_prompt_button.on_click(lambda b: copy_to_prompt_callback(item.sasa))
        output = widgets.Output()
        with output:
            if item.sasa is None:
                print("Solvent Accessible Surface Area (SASA) is not available.")
            else:
                sasa = [s or 0 for s in item.sasa]
                draw_data_array(output, data_array=sasa, cmap="Reds")

        if copy_to_prompt_callback:
            sasa_items.append(
                widgets.VBox(
                    [copy_to_prompt_button, output], layout={"border": "1px solid gray"}
                )
            )
        else:
            sasa_items.append(
                widgets.VBox([output], layout={"border": "1px solid gray"})
            )
    return widgets.VBox(sasa_items)


def create_secondary_structure_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    ss_items = []
    for item in items:
        copy_to_prompt_button = widgets.Button(
            description="Copy to Prompt", disabled=copy_to_prompt_callback is None
        )
        if copy_to_prompt_callback:
            copy_to_prompt_button.on_click(
                lambda b: copy_to_prompt_callback(item.secondary_structure)
            )
        output = widgets.Output()
        with output:
            if item.secondary_structure is None:
                print("Secondary structure is not available.")
            else:
                letter_to_ss3_map = {
                    "C": 0,  # Coil
                    "H": 1,  # Alpha helix
                    "E": 2,  # Beta strand
                }
                secondary_structure = [
                    letter_to_ss3_map.get(s, 0) for s in item.secondary_structure
                ]
                draw_data_array(
                    output,
                    data_array=secondary_structure,
                    categories=[
                        "Coil (C)",
                        "Alpha helix (H)",
                        "Beta strand (E)",
                    ],  # Define appropriate categories
                    category_color_mapping={
                        "Alpha helix (H)": "#77DD77",
                        "Beta strand (E)": "#FF7F7F",
                        "Coil (C)": "#AEC6CF",
                    },
                    highlighted_ranges=[],
                    cmap="Set2",
                )

        if copy_to_prompt_callback:
            ss_items.append(
                widgets.VBox(
                    [copy_to_prompt_button, output], layout={"border": "1px solid gray"}
                )
            )
        else:
            ss_items.append(widgets.VBox([output], layout={"border": "1px solid gray"}))
    return widgets.VBox(ss_items)


def create_structure_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    # Calculate the grid size
    n = len(items)
    grid_size = int(n**0.5)
    if grid_size**2 < n:
        grid_size += 1

    grid = widgets.GridspecLayout(grid_size, grid_size, width="100%")
    for i, item in enumerate(items):
        if item.ptm is None:
            ptm_label = widgets.Label(value="pTM: N/A")
        else:
            ptm_label = widgets.Label(value=f"pTM: {item.ptm.item():.2f}")
        copy_to_prompt_button = widgets.Button(
            description="Copy to Prompt", disabled=copy_to_prompt_callback is None
        )
        if copy_to_prompt_callback:
            copy_to_prompt_button.on_click(
                lambda b: copy_to_prompt_callback(item.coordinates)
            )

        download_pdb_button = create_download_button_from_buffer(
            protein_to_pdb_buffer(item),
            filename=f"generation_{i:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdb",
            description="Download PDB",
        )

        output = widgets.Output()
        if item.coordinates is None:
            with output:
                print("Structure is not available.")
        else:
            cmap = plt.get_cmap("Blues")
            max_confidence = 1.0
            min_confidence = 0.0

            def confidence_to_color(confidence) -> str:
                max_value = cmap.N - 1
                value = int(
                    (confidence - min_confidence)
                    / (max_confidence - min_confidence)
                    * max_value
                )
                color = cmap(value)
                hex_color = mcolors.to_hex(color)
                return hex_color

            if item.plddt is None:
                highlighted_ranges = []
            else:
                plddt: list[float] = item.plddt.tolist()
                highlighted_ranges = [
                    (idx, idx, confidence_to_color(confidence))
                    for idx, confidence in enumerate(plddt)
                ]

            draw_protein_structure(
                output,
                protein_chain=item.to_protein_chain(),
                highlighted_ranges=highlighted_ranges,
            )
        row = i // grid_size
        col = i % grid_size

        if copy_to_prompt_callback:
            header = widgets.HBox(
                [copy_to_prompt_button, download_pdb_button, ptm_label]
            )
        else:
            header = widgets.HBox([download_pdb_button, ptm_label])

        grid[row, col] = widgets.VBox(
            [header, output], layout={"border": "1px solid gray"}
        )
    return grid


def create_function_annotations_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    function_items = []
    for item in items:
        copy_to_prompt_button = widgets.Button(
            description="Copy to Prompt", disabled=copy_to_prompt_callback is None
        )
        if copy_to_prompt_callback:
            copy_to_prompt_button.on_click(
                lambda b: copy_to_prompt_callback(item.function_annotations)
            )
        output = widgets.Output()
        interpro_annotations = (
            [annot for annot in item.function_annotations if "IPR" in annot.label]
            if item.function_annotations is not None
            else []
        )
        if not interpro_annotations:
            with output:
                print("Interpro annotations are not available.")
            function_items.append(
                widgets.VBox(
                    [copy_to_prompt_button, output], layout={"border": "1px solid gray"}
                )
            )
        else:
            image = draw_function_annotations(
                interpro_annotations, sequence_length=len(item)
            )
            if copy_to_prompt_callback:
                content = widgets.VBox(
                    [copy_to_prompt_button, image], layout={"border": "1px solid gray"}
                )
            else:
                content = widgets.VBox([image], layout={"border": "1px solid gray"})
            function_items.append(content)

    return widgets.VBox(function_items)
