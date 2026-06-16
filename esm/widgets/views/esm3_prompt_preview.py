import torch
from ipywidgets import widgets

from esm.sdk.api import ESMProtein, FunctionAnnotation
from esm.utils.constants.esm3 import MASK_STR_SHORT


def coordinates_to_text(coordinates: torch.Tensor | None) -> str:
    if coordinates is None:
        return ""

    non_empty_coordinates = coordinates.isfinite().all(dim=-1).any(dim=-1)
    non_empty_coordinates = non_empty_coordinates.tolist()
    coordinates_text = []
    for value in non_empty_coordinates:
        # Place a checkmark symbol for non-empty coordinates
        if value:
            coordinates_text.append("✓")
        else:
            coordinates_text.append(MASK_STR_SHORT)
    return "".join(coordinates_text)


def sasa_to_text(sasa: list[int | float | None] | None) -> str:
    if sasa is None:
        return ""

    sasa_text = []
    for value in sasa:
        if value is None:
            sasa_text.append(MASK_STR_SHORT)
        elif isinstance(value, float):
            sasa_text.append(f"{value:.2f}")
        else:
            sasa_text.append(str(value))
    return ",".join(sasa_text)


def text_to_sasa(sasa_text: str) -> list[int | float | None] | None:
    if not sasa_text:
        return None

    sasa = []
    for value in sasa_text.split(","):
        if value == MASK_STR_SHORT:
            sasa.append(None)
        else:
            sasa.append(float(value))

    return sasa


def function_annotations_to_text(annotations: list[FunctionAnnotation]) -> str:
    return "\n".join(
        [
            f"[{annotation.start - 1}-{annotation.end - 1}]: {annotation.label}"
            for annotation in annotations
        ]
    )


def create_text_area(
    description, value="", height="100px", width="90%", disabled=False
):
    label = widgets.Label(value=description)
    textarea = widgets.Textarea(
        value=value,
        disabled=disabled,
        layout=widgets.Layout(height=height, width=width),
    )
    return widgets.VBox([label, textarea])


def create_esm3_prompt_preview(protein: ESMProtein) -> widgets.Widget:
    sequence_text = create_text_area(
        "Sequence:", protein.sequence if protein.sequence else ""
    )
    structure_text = create_text_area(
        "Structure:",
        coordinates_to_text(protein.coordinates)
        if protein.coordinates is not None
        else "",
        disabled=True,
    )
    secondary_structure_text = create_text_area(
        "Secondary Structure:",
        protein.secondary_structure if protein.secondary_structure else "",
    )
    sasa_text = create_text_area(
        "Solvent Accessible Surface Area (SASA):",
        sasa_to_text(protein.sasa) if protein.sasa else "",
    )
    function_text = create_text_area(
        "Function:",
        function_annotations_to_text(protein.function_annotations)
        if protein.function_annotations
        else "",
        disabled=True,
    )

    save_changes_button = widgets.Button(
        description="Save Changes",
        disabled=True,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click to save changes to the prompt",
    )

    output = widgets.Output()

    prompt_preview = widgets.VBox(
        [
            sequence_text,
            structure_text,
            secondary_structure_text,
            sasa_text,
            function_text,
            save_changes_button,
            output,
        ],
        layout=widgets.Layout(width="90%"),
    )

    def check_changes(*args, **kwargs):
        output.clear_output()
        sequence_change = protein.sequence != sequence_text.children[1].value
        secondary_structure_change = (
            protein.secondary_structure != secondary_structure_text.children[1].value
        )
        sasa_change = sasa_to_text(protein.sasa) != sasa_text.children[1].value
        if sequence_change or secondary_structure_change or sasa_change:
            save_changes_button.disabled = False

    def save_changes(*args, **kwargs):
        output.clear_output()
        changes = {
            "sequence": sequence_text.children[1].value,
            "secondary_structure": secondary_structure_text.children[1].value,
            "sasa": sasa_text.children[1].value,
        }

        for track, change in changes.items():
            if change:
                if track == "sasa":
                    invalid_length = text_to_sasa(change) != len(protein)
                else:
                    invalid_length = len(change) != len(protein)

                if invalid_length:
                    with output:
                        print(
                            f"Invalid length for {track}. Expected {len(protein)} characters."
                        )
                    return

        protein.sequence = changes["sequence"] if changes["sequence"] else None
        protein.secondary_structure = (
            changes["secondary_structure"] if changes["secondary_structure"] else None
        )
        protein.sasa = text_to_sasa(changes["sasa"]) if changes["sasa"] else None
        save_changes_button.disabled = True
        with output:
            print("Changes saved!")

    sequence_text.children[1].observe(check_changes, names="value")
    secondary_structure_text.children[1].observe(check_changes, names="value")
    sasa_text.children[1].observe(check_changes, names="value")
    save_changes_button.on_click(save_changes)

    return prompt_preview
