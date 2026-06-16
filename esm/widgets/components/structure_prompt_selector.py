from functools import partial
from typing import Callable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from matplotlib.patches import Rectangle

from esm.utils.structure.protein_chain import ProteinChain
from esm.widgets.utils import indexing
from esm.widgets.utils.drawing.draw_protein_structure import draw_protein_structure
from esm.widgets.utils.parsing import convert_range_string_to_list_of_ranges
from esm.widgets.utils.printing import wrapped_print
from esm.widgets.utils.prompting import PromptManager


def create_structure_prompt_selector(
    prompt_manager: PromptManager,
    protein_chain: ProteinChain,
    tag: str,
    with_title: bool = True,
    active_tag_callback: Callable[[], str] | None = None,
) -> widgets.Widget:
    adjacency_matrix = protein_chain.cbeta_contacts(distance_threshold=8)
    size = adjacency_matrix.shape[0]

    min_residue, max_residue = indexing.get_pdb_index_min_max(protein_chain)

    is_active_callback = lambda: (
        active_tag_callback() == tag if active_tag_callback else True
    )

    matrix_output = widgets.Output()
    protein_output = widgets.Output()

    structure_title = widgets.HTML(value="<h1>Structure:</h1>")
    index_option = widgets.RadioButtons(
        options=[indexing.ZERO_INDEX, indexing.PDB_INDEX],
        value=indexing.ZERO_INDEX,
        description="Index: ",
        disabled=False,
    )
    options_ui = widgets.HBox([index_option])
    if with_title:
        header_ui = widgets.VBox([structure_title, options_ui])
    else:
        header_ui = options_ui

    x_slider = widgets.IntRangeSlider(
        value=[0, size - 1],
        min=0,
        max=size - 1,
        step=1,
        description="X",
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=widgets.Layout(width="400px"),
    )
    y_slider = widgets.IntRangeSlider(
        value=[0, size - 1],
        min=0,
        max=size - 1,
        step=1,
        description="Y",
        orientation="vertical",
        readout=True,
        readout_format="d",
        layout=widgets.Layout(height="400px"),
        disabled=True,
    )
    toggle_sync = widgets.Checkbox(
        value=True,
        description="Diagonal",
        disabled=False,
        indent=False,
        tooltip="Sync or Unsync Sliders",
        layout=widgets.Layout(width="75px"),
    )
    slider_link = widgets.dlink((x_slider, "value"), (y_slider, "value"))
    left_ui = widgets.VBox([y_slider, toggle_sync])
    matrix_ui = widgets.VBox([matrix_output, x_slider])
    interactive_ui = widgets.HBox([left_ui, matrix_ui, protein_output])
    error_output = widgets.Output()

    main_ui = widgets.VBox(
        [header_ui, interactive_ui, error_output, prompt_manager.get_selection_ui()]
    )

    contact_map_selection_cache: dict[tuple, tuple] = {}

    def display_matrix_with_highlight(x_range, y_range):
        with matrix_output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(
                adjacency_matrix[::-1, :] > 0,
                cmap="Greys",
                interpolation="none",
                aspect="equal",
            )

            max_y = adjacency_matrix.shape[0]
            for _, (color, selected_ranges, _) in prompt_manager.get_prompts(
                tag=tag
            ).items():
                selected_ranges = tuple(selected_ranges)  # Convert to hashable
                if selected_ranges in contact_map_selection_cache:
                    ((x_start, x_end), (y_start, y_end)) = contact_map_selection_cache[
                        selected_ranges
                    ]
                    rect = Rectangle(
                        (x_start - 0.5, max_y - y_end - 1.5),
                        x_end - x_start + 1,
                        y_end - y_start + 1,
                        linewidth=1,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.2,
                    )
                    ax.add_patch(rect)
                else:
                    for start, end in selected_ranges:
                        y_start, y_end = max_y - end - 1, max_y - start - 1
                        rect = Rectangle(
                            (start - 0.5, y_start - 1.5),
                            end - start + 1,
                            y_end - y_start + 1,
                            linewidth=1,
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.2,
                        )
                        ax.add_patch(rect)

            if not prompt_manager.manual_selection_checkbox.value:
                y_range = (max_y - y_range[1] - 1, max_y - y_range[0] - 1)
                rect = Rectangle(
                    (x_range[0] - 0.5, y_range[0] - 1.5),
                    x_range[1] - x_range[0] + 1,
                    y_range[1] - y_range[0] + 1,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=prompt_manager.get_current_color(),
                    alpha=0.5,
                )
                ax.add_patch(rect)

            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()

    def display_protein():
        highlighted_ranges = []
        for prompt_string, (color, selected_ranges, _) in prompt_manager.get_prompts(
            tag=tag
        ).items():
            for start, end in selected_ranges:
                if indexing.PDB_INDEX_SUFFIX not in prompt_string:
                    start = indexing.zero_index_to_pdb_index(start, protein_chain)
                    end = indexing.zero_index_to_pdb_index(end, protein_chain)
                highlighted_ranges.append((start, end, color))

        if not prompt_manager.manual_selection_checkbox.value:
            selected_ranges = get_selected_residues_in_zero_index()
            selected_ranges = [
                indexing.zero_index_to_pdb_index(r, protein_chain)
                for r in selected_ranges
            ]
            highlighted_ranges.extend(
                [(r, r, prompt_manager.get_current_color()) for r in selected_ranges]
            )
        draw_protein_structure(
            protein_output,
            protein_chain=protein_chain,
            highlighted_ranges=highlighted_ranges,
        )

    def update_section(*args, **kwargs):
        x_range = x_slider.value
        y_range = y_slider.value
        error_output.clear_output()
        if index_option.value == indexing.PDB_INDEX:
            try:
                # Convert to zero index for contact map highlighting
                x_range = indexing.pdb_range_to_zero_range(x_range, protein_chain)
                y_range = indexing.pdb_range_to_zero_range(y_range, protein_chain)
            except Exception as e:
                with error_output:
                    wrapped_print(e)
                return
        display_matrix_with_highlight(x_range, y_range)
        display_protein()

    def toggle_sync_sliders(change):
        if change["new"]:
            slider_link.link()
            y_slider.disabled = True
        else:
            slider_link.unlink()
            y_slider.disabled = False

    def on_index_option_change(change):
        # Note: Value changes in the sliders can trigger a cascade of updates
        # We pause the sync to avoid unnecessary updates
        unset_slider_observe()
        if change["new"] == indexing.ZERO_INDEX:
            new_value = indexing.pdb_range_to_zero_range(x_slider.value, protein_chain)
            new_y_value = indexing.pdb_range_to_zero_range(
                y_slider.value, protein_chain
            )
            x_slider.min = 0
            x_slider.max = size - 1
            x_slider.value = new_value
            y_slider.min = 0
            y_slider.max = size - 1
            y_slider.value = new_y_value
        elif change["new"] == indexing.PDB_INDEX:
            new_value = indexing.zero_range_to_pdb_range(x_slider.value, protein_chain)
            new_y_value = indexing.zero_range_to_pdb_range(
                y_slider.value, protein_chain
            )
            if min_residue > x_slider.max:
                x_slider.max = max_residue
                x_slider.min = min_residue
            else:
                x_slider.min = min_residue
                x_slider.max = max_residue
            if min_residue > y_slider.max:
                y_slider.max = max_residue
                y_slider.min = min_residue
            else:
                y_slider.min = min_residue
                y_slider.max = max_residue
            x_slider.value = new_value
            y_slider.value = new_y_value

        if toggle_sync.value:
            slider_link.link()

        set_slider_observe()
        update_section()

    def toggle_manual_selection(change):
        if change["new"]:
            toggle_sync.disabled = True
            x_slider.disabled = True
            y_slider.disabled = True
        else:
            toggle_sync.disabled = False
            x_slider.disabled = False
            y_slider.disabled = False if not toggle_sync.value else True
        update_section()

    def get_selected_residues_in_zero_index() -> list[int]:
        x_range = x_slider.value
        y_range = y_slider.value

        if index_option.value == indexing.PDB_INDEX:
            # Note: To index the adjacency matrix we use zero index
            x_range = indexing.pdb_range_to_zero_range(x_range, protein_chain)
            y_range = indexing.pdb_range_to_zero_range(y_range, protein_chain)

        contact_map_selection = adjacency_matrix[
            y_range[0] : y_range[1] + 1, x_range[0] : x_range[1] + 1
        ]
        contact_map_selection = contact_map_selection > 0

        contact_residue_pairs = np.argwhere(contact_map_selection)
        contact_residues = list(
            set(contact_residue_pairs[:, 0] + y_range[0]).union(
                set(contact_residue_pairs[:, 1] + x_range[0])
            )
        )
        return sorted(contact_residues)

    def residue_list_to_list_of_ranges(residues: list[int]) -> list[tuple[int, int]]:
        ranges = []
        start = end = residues[0]
        for idx in residues[1:]:
            if idx == end + 1:
                end = idx
            else:
                ranges.append((start, end))
                start = end = idx
        ranges.append((start, end))
        return ranges

    def range_to_structure_motif(
        range: tuple[int, int], convert_from_pdb_index: bool
    ) -> torch.Tensor:
        if convert_from_pdb_index:
            range = indexing.pdb_range_to_zero_range(range, protein_chain)

        coordinates, *_ = protein_chain.to_structure_encoder_inputs()
        coordinates = coordinates.squeeze(dim=0)
        values = coordinates[range[0] : range[1] + 1]

        if len(values) != range[1] - range[0] + 1:
            raise IndexError(
                "Values in the range do not match the expected number of residues. "
                f"Expected: {range[1] - range[0] + 1}, Found: {len(values)}\n"
                "If this is a PDB index issue, please use zero index instead."
            )
        return values

    def handle_add_button_click(_):
        if is_active_callback():
            if prompt_manager.manual_selection_checkbox.value:
                selected_ranges = convert_range_string_to_list_of_ranges(
                    prompt_manager.manual_input.value
                )
            else:
                try:
                    selected_ranges = residue_list_to_list_of_ranges(
                        get_selected_residues_in_zero_index()
                    )
                    if index_option.value == indexing.PDB_INDEX:
                        # Note: The contact map cache is in zero index
                        x_range = indexing.pdb_range_to_zero_range(
                            x_slider.value, protein_chain
                        )
                        y_range = indexing.pdb_range_to_zero_range(
                            y_slider.value, protein_chain
                        )
                    else:
                        x_range = x_slider.value
                        y_range = y_slider.value
                except Exception as e:
                    with error_output:
                        wrapped_print(e)
                    return

                contact_map_selection_cache[tuple(selected_ranges)] = (x_range, y_range)

                if index_option.value == indexing.PDB_INDEX:
                    # Convert back to PDB index
                    selected_ranges = [
                        indexing.zero_range_to_pdb_range(r, protein_chain)
                        for r in selected_ranges
                    ]

            prompt_manager.add_entry(
                selected_ranges,
                get_value_from_range_callback=partial(
                    range_to_structure_motif,
                    convert_from_pdb_index=index_option.value == indexing.PDB_INDEX,
                ),
                tag=tag,
                indexing_type=indexing.PDB_INDEX
                if index_option.value == indexing.PDB_INDEX
                else indexing.ZERO_INDEX,
            )
            update_section()

    def set_slider_observe():
        # Observe changes in sliders to update visualization
        x_slider.observe(update_section, names="value")
        y_slider.observe(update_section, names="value")

    def unset_slider_observe():
        x_slider.unobserve_all()
        y_slider.unobserve_all()

    # Start with sliders synced
    slider_link.link()

    # Set all the observers
    set_slider_observe()
    toggle_sync.observe(toggle_sync_sliders, names="value")
    index_option.observe(on_index_option_change, names="value")

    # Whether to enable manual selection
    prompt_manager.manual_selection_checkbox.observe(
        toggle_manual_selection, names="value"
    )
    prompt_manager.add_button.on_click(handle_add_button_click)
    prompt_manager.register_delete_callback(update_section)

    display_matrix_with_highlight(x_slider.value, y_slider.value)
    display_protein()

    return main_ui
