from collections import defaultdict
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import torch
from ipywidgets import widgets

from esm.sdk.api import ESMProtein, FunctionAnnotation
from esm.utils import encoding
from esm.widgets.utils import indexing
from esm.widgets.utils.drawing.colors import rgba_tuple_to_hex
from esm.widgets.utils.drawing.draw_category_array import draw_data_array
from esm.widgets.utils.printing import wrapped_print


class PromptManagerCollection:
    def __init__(self, prompt_length):
        self.function_annotations: dict[tuple, FunctionAnnotation] = {}
        self.prompt_length = prompt_length
        self.sequence_prompt_manager = PromptManager(prompt_length)
        self.structure_prompt_manager = PromptManager(
            prompt_length, allow_multiple_tags=False
        )
        self.secondary_structure_prompt_manager = PromptManager(prompt_length)
        self.sasa_prompt_manager = PromptManager(prompt_length)

    def reset_all_handlers(self):
        self.sequence_prompt_manager.reset_handlers()
        self.structure_prompt_manager.reset_handlers()
        self.secondary_structure_prompt_manager.reset_handlers()
        self.sasa_prompt_manager.reset_handlers()

    def compile(self) -> ESMProtein:
        return ESMProtein(
            sequence=self._compile_sequence_prompts(),
            secondary_structure=self._compile_secondary_structure_prompts(),
            sasa=self._compile_sasa_prompts(),  # type: ignore
            function_annotations=list(self.function_annotations.values()),
            coordinates=self._compile_structure_prompts(),
        )

    def add_function_annotation(self, annotation: FunctionAnnotation):
        key = (annotation.label, annotation.start, annotation.end)
        self.function_annotations[key] = annotation

    def delete_function_annotation(self, annotation: FunctionAnnotation):
        key = (annotation.label, annotation.start, annotation.end)
        if key in self.function_annotations:
            self.function_annotations.pop(key)

    def _compile_sequence_prompts(self) -> str:
        sequence = list(encoding.get_default_sequence(self.prompt_length))
        sequence_prompts = self.sequence_prompt_manager.get_prompts()
        for prompt_str, (_, _, values) in sequence_prompts.items():
            target_ranges = self.sequence_prompt_manager.prompt_to_target_ranges[
                prompt_str
            ]
            for (start, end), value in zip(target_ranges, values):
                assert len(value) == end - start + 1
                for pos, char in enumerate(value):
                    sequence[start + pos] = char

        return "".join(sequence)

    def _compile_structure_prompts(self) -> torch.Tensor | None:
        structure_prompts = self.structure_prompt_manager.get_prompts()

        if len(structure_prompts) == 0:
            return None

        coordinates = torch.zeros((self.prompt_length, 37, 3))
        coordinates = torch.fill(coordinates, torch.nan)

        for prompt_str, (_, _, values) in structure_prompts.items():
            target_ranges = self.structure_prompt_manager.prompt_to_target_ranges[
                prompt_str
            ]
            for (start, end), value in zip(target_ranges, values):
                assert len(value) == end - start + 1
                coordinates[start : end + 1] = value

        return coordinates

    def _compile_secondary_structure_prompts(self) -> str | None:
        secondary_structure_prompts = (
            self.secondary_structure_prompt_manager.get_prompts()
        )

        if len(secondary_structure_prompts) == 0:
            return None

        secondary_structure = list(
            encoding.get_default_secondary_structure(self.prompt_length)
        )
        for prompt_str, (_, _, values) in secondary_structure_prompts.items():
            target_ranges = (
                self.secondary_structure_prompt_manager.prompt_to_target_ranges[
                    prompt_str
                ]
            )
            for (start, end), value in zip(target_ranges, values):
                assert len(value) == end - start + 1
                for pos, char in enumerate(value):
                    secondary_structure[start + pos] = char

        return "".join(secondary_structure)

    def _compile_sasa_prompts(self) -> Sequence[float | str | None] | None:
        sasa_prompts = self.sasa_prompt_manager.get_prompts()

        if len(sasa_prompts) == 0:
            return None

        sasa = list(encoding.get_default_sasa(self.prompt_length))
        for prompt_str, (_, _, values) in sasa_prompts.items():
            target_ranges = self.sasa_prompt_manager.prompt_to_target_ranges[prompt_str]
            for (start, end), value in zip(target_ranges, values):
                sasa[start : end + 1] = value
        return sasa


class PromptManager:
    def __init__(
        self,
        prompt_length: int,
        allow_multiple_tags: bool = True,
        allow_repeated_prompts: bool = True,
    ):
        self.prompt_length = prompt_length
        self.prompts: dict[
            str, tuple[Any, Sequence[tuple[int, int]], Sequence[Any]]
        ] = {}
        self.current_selection = 0
        self.tag_to_prompts: defaultdict[str, list[str]] = defaultdict(list)
        self.prompt_to_target_ranges: dict[str, Sequence[tuple]] = {}
        self.allow_multiple_tags = allow_multiple_tags
        self.allow_repeated_prompts = allow_repeated_prompts

        self.target_position_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=prompt_length - 1,
            step=1,
            description="Target Start Position in Prompt:",
            continuous_update=False,
            style={"description_width": "initial"},
            readout=True,
            readout_format="d",
            layout=widgets.Layout(width="600px"),
        )
        self.manual_selection_checkbox = widgets.Checkbox(
            value=False,
            description="Manual selection",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="150px"),
        )

        self.manual_input_label = widgets.Label(value="Residues from Source Protein:")
        self.manual_input = widgets.Text(disabled=True)

        self.add_button = widgets.Button(description="Add To Prompt")
        self.entries_box = widgets.VBox()
        self.output = widgets.Output()
        self.error_output = widgets.Output()

        self.manual_selection_checkbox.observe(
            self.toggle_manual_selection, names="value"
        )
        self.target_position_slider.observe(self.redraw, names="value")

        self.selection_ui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.manual_input_label,
                        self.manual_input,
                        self.manual_selection_checkbox,
                    ]
                ),
                self.target_position_slider,
                self.add_button,
                self.error_output,
                self.output,
                self.entries_box,
            ]
        )

        self.add_button.on_click(self.redraw)

        self.delete_callbacks: list[Callable[[], None]] = []

        self.redraw()

    def redraw(self, change=None):
        categories = ["Mask (-)"]
        color_map = {"Mask (-)": "white"}
        data_array = [0] * self.prompt_length
        for prompt_str, *_ in self.prompts.items():
            color, _, _ = self.prompts[prompt_str]
            ranges = self.prompt_to_target_ranges[prompt_str]
            color_map[prompt_str] = color
            categories.append(prompt_str)
            for start, end in ranges:
                data_array[start : end + 1] = [categories.index(prompt_str)] * (
                    end - start + 1
                )
        draw_data_array(
            self.output,
            data_array=data_array,
            categories=categories,
            category_color_mapping=color_map,
            highlighted_ranges=[
                (
                    self.target_position_slider.value,
                    self.target_position_slider.value,
                    self.get_current_color(),
                )
            ],
            use_legend=False,
        )

    def toggle_manual_selection(self, change):
        self.manual_input.disabled = not self.manual_selection_checkbox.value

    def validate_and_transform_ranges(
        self, ranges: Sequence[tuple[int, int]]
    ) -> Sequence[tuple[int, int]]:
        if len(ranges) == 0:
            raise ValueError("No ranges selected")

        ranges = sorted(ranges, key=lambda x: x[0])

        target_start_pos = self.target_position_slider.value
        reference_start_pos = ranges[0][0]

        new_ranges = []
        for start, end in ranges:
            length = end - start + 1
            new_start = start - reference_start_pos + target_start_pos
            new_end = new_start + length - 1
            if new_end >= self.prompt_length:
                raise ValueError(f"Range(s) {ranges} too long to fit in the prompt")
            new_ranges.append((new_start, new_end))

        return new_ranges

    def add_entry(
        self,
        selected_ranges: Sequence[tuple[int, int]],
        tag: str,
        get_value_from_range_callback: Callable[[tuple[int, int]], Any] | None = None,
        indexing_type: str = indexing.ZERO_INDEX,
    ):
        try:
            self.error_output.clear_output()
            label_color = self.get_current_color()
            target_ranges = self.validate_and_transform_ranges(selected_ranges)
            self.validate_unique_tag(tag)
            range_string = self.add_prompt(
                tag,
                label_color,
                selected_ranges,
                target_ranges=target_ranges,
                get_value_from_range_callback=get_value_from_range_callback,
                indexing_type=indexing_type,
            )
            self.add_entry_to_ui(range_string)
            self.redraw()
        except ValueError as e:
            with self.error_output:
                wrapped_print(e)

    def add_entry_to_ui(self, range_string):
        label_color, selected_ranges, _ = self.prompts[range_string]
        entry_button = widgets.Button(description="Delete")
        entry_label = widgets.HTML(
            value=(
                f'<div style="display: inline-block; width: 10px; height: 10px; background-color:{label_color}; margin-right: 5px;"></div>'
                f"{range_string}"
            )
        )
        entry_label.tag = range_string
        entry_container = widgets.HBox([entry_button, entry_label])

        def delete_entry(b):
            self.entries_box.children = [
                w for w in self.entries_box.children if w != entry_container
            ]
            self.delete_prompt(entry_label.tag)
            self.redraw()
            for callback in self.delete_callbacks:
                callback()

        entry_button.on_click(delete_entry)
        self.entries_box.children += (entry_container,)

    def get_selection_ui(self):
        return self.selection_ui

    def reset_handlers(self):
        self.add_button._click_handlers.callbacks = []
        self.delete_callbacks = []

    def register_delete_callback(self, callback: Callable[[], None]):
        self.delete_callbacks.append(callback)

    def add_prompt(
        self,
        tag: str,
        label_color: Any,
        selected_ranges: Sequence[tuple[int, int]],
        target_ranges: Sequence[tuple[int, int]],
        get_value_from_range_callback: Callable[[tuple[int, int]], Any] | None = None,
        indexing_type: str = indexing.ZERO_INDEX,
    ) -> str:
        self.validate_range_overlap(target_ranges)
        range_string = ",".join([f"{start}-{end}" for start, end in selected_ranges])
        target_range_string = ",".join(
            [f"{start}-{end}" for start, end in target_ranges]
        )
        range_string = f"{tag} Source:{range_string} Target:{target_range_string}"
        if indexing_type != indexing.ZERO_INDEX:
            if indexing_type == indexing.PDB_INDEX:
                range_string += indexing.PDB_INDEX_SUFFIX
            else:
                raise ValueError(
                    f"Invalid indexing type {indexing_type}, must be one of {indexing.ZERO_INDEX}, {indexing.PDB_INDEX}"
                )

        prompt_source = range_string.split(" Target:")[0]
        if prompt_source not in [
            prompt.split(" Target:")[0] for prompt in self.prompts
        ]:
            if get_value_from_range_callback:
                values = [
                    get_value_from_range_callback(range) for range in selected_ranges
                ]
            else:
                values = []
            self.prompts[range_string] = (label_color, selected_ranges, values)

            self.current_selection += 1

            self.tag_to_prompts[tag].append(range_string)
            self.prompt_to_target_ranges[range_string] = target_ranges
        else:
            if not self.allow_repeated_prompts:
                raise ValueError(
                    f"Prompt {prompt_source} already exists, and repeated prompts are not allowed"
                )
            # Use previous color for the same prompt source
            old_prompts = [
                prompt
                for prompt in self.prompts.keys()
                if prompt.startswith(prompt_source)
            ]
            old_color, _, old_values = self.prompts[old_prompts[0]]
            self.prompts[range_string] = (old_color, selected_ranges, old_values)
            self.tag_to_prompts[tag].append(range_string)
            self.prompt_to_target_ranges[range_string] = target_ranges

        return range_string

    def delete_prompt(self, range_string):
        if range_string in self.prompts:
            self.prompts.pop(range_string)
            self.prompt_to_target_ranges.pop(range_string)
            for tag, prompts in self.tag_to_prompts.items():
                if range_string in prompts:
                    prompts.remove(range_string)

    def validate_unique_tag(self, ref_tag: str):
        other_tags = [
            tag
            for tag, prompts in self.tag_to_prompts.items()
            if tag != ref_tag and len(prompts) > 0
        ]
        if self.allow_multiple_tags:
            return
        if len(other_tags) > 0:
            raise ValueError(
                f"Only one protein tag allowed, but found prompts with other protein tags {other_tags}"
            )

    def validate_range_overlap(self, target_ranges: Sequence[tuple[int, int]]):
        for start, end in target_ranges:
            for prompt_str, range_list in self.prompt_to_target_ranges.items():
                for range_start, range_end in range_list:
                    if start <= range_end and end >= range_start:
                        raise ValueError(
                            f"Target range {start}-{end} overlaps with '{prompt_str}' at range {range_start}-{range_end}"
                        )

    def get_prompts(
        self, tag: str | None = None
    ) -> dict[str, tuple[Any, Sequence[tuple], Sequence[Any]]]:
        prompts = {**self.prompts}
        if tag is None:
            return prompts
        else:
            return {key: prompts[key] for key in self.tag_to_prompts[tag]}

    def get_current_color(self):
        tab10 = plt.get_cmap("tab10")
        rgba = tab10(self.current_selection % tab10.N)
        rgb = rgba[:3]
        return rgba_tuple_to_hex(rgb)
