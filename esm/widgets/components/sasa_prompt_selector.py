from typing import Any, Callable, Sequence

import ipywidgets as widgets

from esm.utils.structure.protein_chain import ProteinChain
from esm.widgets.utils.drawing.colors import hex_to_rgba_tuple, rgba_tuple_to_hex
from esm.widgets.utils.drawing.draw_category_array import draw_data_array
from esm.widgets.utils.parsing import convert_range_string_to_list_of_ranges
from esm.widgets.utils.prompting import PromptManager


def create_sasa_prompt_selector(
    prompt_manager: PromptManager,
    tag: str,
    *,
    input_array: Sequence[float] | None = None,
    protein_chain: ProteinChain | None = None,
    with_title: bool = True,
    active_tag_callback: Callable[[], str] | None = None,
) -> widgets.Widget:
    is_active_callback = lambda: (
        active_tag_callback() == tag if active_tag_callback else True
    )

    if input_array is None:
        if protein_chain is not None:
            input_array = get_sasa(protein_chain)
        else:
            raise ValueError("Either input_array or protein_chain must be provided.")

    range_slider = widgets.IntRangeSlider(
        value=[0, 2],
        min=0,
        max=len(input_array) - 1,
        step=1,
        description="Range:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="50%"),
    )
    output = widgets.Output()

    highlighted_ranges = []

    def update_highlighted_ranges() -> list[tuple[int, int, Any]]:
        nonlocal highlighted_ranges

        def _lower_alpha(hex_color: str, alpha: float) -> str:
            r, g, b, _ = hex_to_rgba_tuple(hex_color)
            return rgba_tuple_to_hex((r, g, b, alpha))

        highlighted_ranges = [
            (start, end, _lower_alpha(color, alpha=0.6))
            for _, (color, ranges, _) in prompt_manager.get_prompts(tag=tag).items()
            for start, end in ranges
        ]
        # Add current slider range to highlighted ranges
        range_slider_value = range_slider.value
        if range_slider_value:
            highlighted_ranges.insert(
                0,
                (
                    range_slider_value[0],
                    range_slider_value[1],
                    _lower_alpha(prompt_manager.get_current_color(), alpha=0.8),
                ),
            )
        return highlighted_ranges

    def redraw(highlighted_ranges):
        draw_data_array(
            output,
            data_array=input_array,
            category_color_mapping={},
            highlighted_ranges=highlighted_ranges,
            cmap="Reds",
        )

    redraw(highlighted_ranges)

    def range_to_sasa_motif(range: tuple[int, int]) -> Sequence[float]:
        return input_array[range[0] : range[1] + 1]

    def add_entry(_):
        if is_active_callback():
            if prompt_manager.manual_selection_checkbox.value:
                range_string = prompt_manager.manual_input.value
                selected_ranges = convert_range_string_to_list_of_ranges(range_string)
            else:
                start, end = range_slider.value
                selected_ranges = [(start, end)]

            prompt_manager.add_entry(
                selected_ranges,
                tag=tag,
                get_value_from_range_callback=range_to_sasa_motif,
            )
            update_visualizer()

    def toggle_manual_selection(change):
        if change["new"]:
            prompt_manager.manual_input.disabled = False
            range_slider.disabled = True
            highlighted_ranges = update_highlighted_ranges()
            highlighted_ranges.pop()  # Remove current slider range
            redraw(highlighted_ranges)
        else:
            prompt_manager.manual_input.disabled = True
            range_slider.disabled = False
            update_visualizer()

    def update_visualizer(*args, **kwargs):
        highlighted_ranges = update_highlighted_ranges()
        redraw(highlighted_ranges)

    prompt_manager.add_button.on_click(add_entry)
    prompt_manager.manual_selection_checkbox.observe(
        toggle_manual_selection, names="value"
    )
    prompt_manager.register_delete_callback(update_visualizer)

    range_slider.observe(update_visualizer, names="value")
    main_ui = widgets.VBox([range_slider, output, prompt_manager.get_selection_ui()])

    if with_title:
        heading = widgets.HTML(value="<h1>SASA Prompt Selector:</h1>")
        parent_layout = widgets.VBox([heading, main_ui])
        return parent_layout
    else:
        return main_ui


def get_sasa(protein_chain: ProteinChain) -> Sequence[float]:
    sasa_values = protein_chain.sasa().tolist()
    return sasa_values
