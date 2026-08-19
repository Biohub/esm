from ipywidgets import widgets

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    GenerationConfig,
)
from esm.widgets.components.results_visualizer import (
    create_results_visualizer,
)
from esm.widgets.utils.printing import wrapped_print
from esm.widgets.utils.protein_import import ProteinImporter


def create_inverse_folding_ui(client: ESM3InferenceClient) -> widgets.Widget:
    # Alow a single protein and immediately load it from workspace
    protein_importer = ProteinImporter(max_proteins=1, autoload=True)
    output = widgets.Output()
    inverse_folding_ui = widgets.VBox([protein_importer.importer_ui, output])

    inverse_fold_button = widgets.Button(
        description="Inverse Fold",
        disabled=True,
        tooltip="Click to predict the protein sequence from the structure",
        style={"button_color": "lightgreen"},
    )

    def get_protein() -> ESMProtein:
        [first_protein] = protein_importer.protein_list
        protein_id, protein_chain = first_protein
        protein = ESMProtein.from_protein_chain(protein_chain)

        # NOTE: We ignore all properties except structure
        protein.sequence = None
        protein.secondary_structure = None
        protein.sasa = None
        protein.function_annotations = None
        return protein

    def on_new_protein(_):
        is_protein = len(protein_importer.protein_list) > 0
        inverse_fold_button.disabled = not is_protein
        inverse_folding_ui.children = [
            protein_importer.importer_ui,
            inverse_fold_button,
            output,
        ]

    def validate_inverse_fold(_):
        if len(protein_importer.protein_list) == 0:
            inverse_fold_button.disabled = True
        else:
            inverse_fold_button.disabled = False

    def on_click_inverse_fold(_):
        try:
            # Reset the output and results
            output.clear_output()
            inverse_folding_ui.children = [
                protein_importer.importer_ui,
                inverse_fold_button,
                output,
            ]
            # Predict the protein's sequence
            protein = get_protein()
            with output:
                print("Predicting the protein sequence from the structure...")
                protein = client.generate(
                    input=protein,
                    config=GenerationConfig(track="sequence", num_steps=1),
                )
                if isinstance(protein, ESMProteinError):
                    wrapped_print(f"Protein Error: {protein.error_msg}")
                elif isinstance(protein, ESMProtein):
                    sequence_results = create_results_visualizer(
                        modality="sequence",
                        samples=[protein],
                        items_per_page=1,
                        include_title=False,
                    )
                    output.clear_output()
                    inverse_folding_ui.children = [
                        protein_importer.importer_ui,
                        inverse_fold_button,
                        sequence_results,
                    ]
        except Exception as e:
            with output:
                wrapped_print(e)

    inverse_fold_button.on_click(on_click_inverse_fold)
    protein_importer.entries_box.observe(on_new_protein, names="children")
    protein_importer.register_delete_callback(lambda: validate_inverse_fold(None))

    return inverse_folding_ui
