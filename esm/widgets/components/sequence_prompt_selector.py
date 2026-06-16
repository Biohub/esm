from typing import Callable

import ipywidgets as widgets

from esm.widgets.utils.drawing.colors import (
    hex_to_rgba_tuple,
    rgba_tuple_to_rgba_html_string,
)
from esm.widgets.utils.parsing import convert_range_string_to_list_of_ranges
from esm.widgets.utils.prompting import PromptManager


def create_sequence_prompt_selector(
    prompt_manager: PromptManager,
    tag: str,
    full_sequence: str,
    line_length=120,
    with_title: bool = True,
    active_tag_callback: Callable[[], str] | None = None,
) -> widgets.Widget:
    sequence_length = len(full_sequence)

    is_active_callback = lambda: (
        active_tag_callback() == tag if active_tag_callback else True
    )

    range_slider = widgets.IntRangeSlider(
        value=[0, sequence_length - 1],
        min=0,
        max=sequence_length - 1,
        step=1,
        description="Crop Range",
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=widgets.Layout(width="600px"),
    )

    mask_option = widgets.RadioButtons(
        options=["Custom range", "Fully masked"],
        value="Custom range",
        description="Mask Option:",
        disabled=False,
    )

    output = widgets.HTML(layout=widgets.Layout(width="600px", white_space="pre-wrap"))

    def toggle_mask_option(change):
        if mask_option.value == "Fully masked":
            range_slider.disabled = True
            if is_active_callback():
                prompt_manager.add_button.disabled = True
        else:
            range_slider.disabled = False
            if is_active_callback():
                prompt_manager.add_button.disabled = False
        update_sequence()

    def toggle_manual_selection(change):
        if change["new"]:
            prompt_manager.manual_input.disabled = False
            range_slider.disabled = True
        else:
            prompt_manager.manual_input.disabled = True
            range_slider.disabled = False

    def update_sequence(change=None):
        highlighted_sequence = list(full_sequence)
        ranges = []
        for _, (color, rs, _) in prompt_manager.get_prompts(tag=tag).items():
            for start, end in rs:
                ranges.append((start, end, color))

        if ranges:
            highlighted_sequence = apply_highlighting(full_sequence, ranges)
        else:
            highlighted_sequence = add_line_breaks(full_sequence)

        current_range = ["_"] * sequence_length
        start, end = range_slider.value
        current_range[start : end + 1] = full_sequence[start : end + 1]

        current_color = prompt_manager.get_current_color()
        current_range_html = f"<b>Current Selection:</b><p style='color:{current_color}; font-family:monospace;'>{add_line_breaks(''.join(current_range))}</p><br>"
        highlighted_sequence_html = f"<b>Selected:</b><p style='font-family:monospace;'>{highlighted_sequence}<br>{current_range_html}</p>"
        output.value = highlighted_sequence_html

    def add_line_breaks(sequence):
        return "<br>".join(
            [
                sequence[i : i + line_length]
                for i in range(0, len(sequence), line_length)
            ]
        )

    def apply_highlighting(sequence, ranges):
        lines = [
            sequence[i : i + line_length] for i in range(0, len(sequence), line_length)
        ]

        highlighted_lines = []
        for line_idx, line in enumerate(lines):
            line_start = line_idx * line_length
            line_end = line_start + len(line)
            highlighted_line = list(line)
            all_line_ranges = [
                (
                    max(start, line_start) - line_start,
                    min(end, line_end - 1) - line_start,
                    color,
                )
                for start, end, color in ranges
                if start < line_end and end >= line_start
            ]
            for i in range(len(highlighted_line)):
                span_layers = []
                for start, end, color in all_line_ranges:
                    if start <= i <= end:
                        span_layers.append(color)
                if span_layers:
                    combined_color = span_layers[-1]
                    r, g, b, a = hex_to_rgba_tuple(combined_color)
                    a = 0.5  # Set alpha to 0.5
                    combined_color = rgba_tuple_to_rgba_html_string((r, g, b, a))
                    highlighted_line[i] = (
                        f'<span style="background-color:{combined_color}">{highlighted_line[i]}</span>'
                    )
            highlighted_lines.append("".join(highlighted_line))

        return "<br>".join(highlighted_lines)

    range_slider.observe(update_sequence, names="value")
    mask_option.observe(toggle_mask_option, names="value")

    def range_to_sequence_motif(range: tuple[int, int]) -> str:
        return full_sequence[range[0] : range[1] + 1]

    def handle_add_button_click(_):
        if is_active_callback():
            if prompt_manager.manual_selection_checkbox.value:
                selected_ranges = convert_range_string_to_list_of_ranges(
                    prompt_manager.manual_input.value
                )
            else:
                selected_ranges = [range_slider.value]
            prompt_manager.add_entry(
                selected_ranges,
                tag=tag,
                get_value_from_range_callback=range_to_sequence_motif,
            )
            update_sequence()

    prompt_manager.add_button.on_click(lambda x: handle_add_button_click(""))
    prompt_manager.register_delete_callback(update_sequence)
    prompt_manager.manual_selection_checkbox.observe(
        toggle_manual_selection, names="value"
    )

    update_sequence()

    if with_title:
        return widgets.VBox(
            [
                widgets.HTML(value="<h1>Sequence:</h1>"),
                mask_option,
                range_slider,
                output,
                prompt_manager.get_selection_ui(),
            ]
        )
    else:
        return widgets.VBox(
            [mask_option, range_slider, output, prompt_manager.get_selection_ui()]
        )
