from typing import Any, Literal

from ipywidgets import widgets

from esm.sdk.api import ESM3InferenceClient, ESMProtein
from esm.utils.constants import esm3 as C
from esm.widgets.components.function_annotator import (
    create_function_annotator,
)
from esm.widgets.utils.prompting import PromptManagerCollection
from esm.widgets.utils.protein_import import ProteinImporter
from esm.widgets.views.esm3_generation_launcher import (
    create_esm3_generation_launcher,
)
from esm.widgets.views.esm3_prompt_preview import (
    create_esm3_prompt_preview,
)
from esm.widgets.views.esm3_prompt_selector import (
    create_esm3_prompt_selector,
)


def create_generation_ui(
    client: ESM3InferenceClient | None = None, forge_token: str = ""
) -> widgets.Widget:
    # Specify Protein Length
    protein_length_input = widgets.IntText(
        value=100,
        description="Length:",
        disabled=False,
        layout=widgets.Layout(width="200px"),
    )
    protein_length_confirm_button = widgets.Button(
        description="Confirm",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click to confirm the protein length",
    )
    output = widgets.Output()
    protein_length_ui = widgets.VBox(
        [
            widgets.HTML(value="<h3>Specify Prompt Length:</h3>"),
            widgets.HBox([protein_length_input, protein_length_confirm_button]),
            output,
        ]
    )
    loading_ui = widgets.HTML(value="<h3>Loading...</h3>")

    # Import Proteins
    protein_importer = ProteinImporter()
    protein_importer_ui = widgets.VBox(
        [
            widgets.HTML(value="<h3>Start from One or More Reference Proteins:</h3>"),
            protein_importer.importer_ui,
        ]
    )

    # Select Motifs
    prompt_manager_collection = PromptManagerCollection(protein_length_input.value)
    esm3_selector = create_esm3_prompt_selector(
        prompt_manager_collection, protein_importer=protein_importer
    )
    selector_title = widgets.HTML(
        value="<h3>Select Motifs from Reference Protein(s):</h3>"
    )
    selector_ui = widgets.VBox([selector_title, esm3_selector])

    # Function Annotations
    function_annotator_title = widgets.HTML(
        value="<h3>Add Function Annotations to the Prompt:</h3>"
    )
    function_annotator = create_function_annotator(
        protein_length_input.value,
        add_annotation_callback=prompt_manager_collection.add_function_annotation,
        delete_annotation_callback=prompt_manager_collection.delete_function_annotation,
    )
    function_annotator_ui = widgets.VBox([function_annotator_title, function_annotator])

    # Compile Prompt
    protein: ESMProtein | None = None

    compile_title = widgets.HTML(value="<h3>Compile Prompt:</h3>")
    compile_button = widgets.Button(
        description="Compile Prompt",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click to compile the selected motifs into an ESMProtein",
    )
    compile_ui = widgets.VBox([compile_title, compile_button])

    # Main UI
    prompt_ui = widgets.VBox(
        [protein_importer_ui, protein_length_ui, function_annotator_ui, compile_ui]
    )

    # Callbacks
    def update_selector(*args, **kwargs):
        nonlocal prompt_manager_collection

        validate_protein_length()

        prompt_manager_collection = PromptManagerCollection(protein_length_input.value)
        function_annotator = create_function_annotator(
            protein_length_input.value,
            add_annotation_callback=prompt_manager_collection.add_function_annotation,
            delete_annotation_callback=prompt_manager_collection.delete_function_annotation,
        )
        function_annotator_ui.children = [function_annotator_title, function_annotator]

        if len(protein_importer.protein_list) == 0:
            prompt_ui.children = [
                protein_importer_ui,
                protein_length_ui,
                function_annotator_ui,
                compile_ui,
            ]
        elif len(protein_importer.protein_list) > 0:
            prompt_ui.children = [
                protein_importer_ui,
                protein_length_ui,
                function_annotator_ui,
                loading_ui,
            ]
            esm3_selector_ui = create_esm3_prompt_selector(
                prompt_manager_collection, protein_importer=protein_importer
            )
            selector_ui.children = [selector_title, esm3_selector_ui]
            prompt_ui.children = [
                protein_importer_ui,
                protein_length_ui,
                function_annotator_ui,
                selector_ui,
                compile_ui,
            ]

    def copy_to_prompt_callback(
        modality: Literal[
            "sequence", "coordinates", "secondary_structure", "sasa", "function"
        ],
        value: Any,
    ):
        nonlocal protein
        if protein is not None:
            if modality == "sequence":
                value = [
                    C.MASK_STR_SHORT if x == C.SEQUENCE_MASK_TOKEN else x for x in value
                ]
                value = "".join(value)

            elif modality == "secondary_structure":
                value = [C.MASK_STR_SHORT if x == C.SS8_PAD_TOKEN else x for x in value]
                value = "".join(value)

            setattr(protein, modality, value)
            prompt_preview = create_esm3_prompt_preview(protein)
            preview_ui = widgets.VBox(
                [
                    widgets.HTML(value="<h3>Preview and Edit Prompt:</h3>"),
                    prompt_preview,
                ]
            )
            generation_launcher = create_esm3_generation_launcher(
                protein=protein,
                client=client,
                forge_token=forge_token,
                copy_to_prompt_callback=copy_to_prompt_callback,
            )
            generation_launcher_ui = widgets.VBox(
                [widgets.HTML(value="<h3>Generation Config:</h3>"), generation_launcher]
            )

            if len(protein_importer.protein_list) > 0:
                prompt_ui.children = [
                    protein_importer_ui,
                    protein_length_ui,
                    function_annotator_ui,
                    selector_ui,
                    compile_ui,
                    preview_ui,
                    generation_launcher_ui,
                ]
            else:
                prompt_ui.children = [
                    protein_importer_ui,
                    protein_length_ui,
                    function_annotator_ui,
                    compile_ui,
                    preview_ui,
                    generation_launcher_ui,
                ]

    def on_compile(*args, **kwargs):
        nonlocal protein
        prompt_ui.children = [protein_importer_ui, protein_length_ui, loading_ui]
        protein = prompt_manager_collection.compile()
        prompt_preview = create_esm3_prompt_preview(protein)
        preview_ui = widgets.VBox(
            [widgets.HTML(value="<h3>Preview and Edit Prompt:</h3>"), prompt_preview]
        )
        generation_launcher = create_esm3_generation_launcher(
            protein=protein,
            client=client,
            forge_token=forge_token,
            copy_to_prompt_callback=copy_to_prompt_callback,
        )
        generation_launcher_ui = widgets.VBox(
            [widgets.HTML(value="<h3>Generation Config:</h3>"), generation_launcher]
        )

        if len(protein_importer.protein_list) > 0:
            prompt_ui.children = [
                protein_importer_ui,
                protein_length_ui,
                function_annotator_ui,
                selector_ui,
                compile_ui,
                preview_ui,
                generation_launcher_ui,
            ]
        else:
            prompt_ui.children = [
                protein_importer_ui,
                protein_length_ui,
                function_annotator_ui,
                compile_ui,
                preview_ui,
                generation_launcher_ui,
            ]

    def validate_protein_length(*args, **kwargs):
        output.clear_output()
        with output:
            if protein_length_input.value < 1:
                print("Protein length must be at least 1.")
            elif protein_length_input.value > 2048:
                print("Protein length must be at most 2048.")

    def on_confirm(*args, **kwargs):
        validate_protein_length()
        with output:
            print("Protein length set to:", protein_length_input.value)

    protein_length_confirm_button.on_click(update_selector)
    protein_length_confirm_button.on_click(on_confirm)
    protein_importer.pdb_id_add_button.on_click(update_selector)
    protein_importer.pdb_uploader.observe(update_selector, "value")
    protein_importer.register_delete_callback(update_selector)
    compile_button.on_click(on_compile)

    return prompt_ui
