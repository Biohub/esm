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


def create_prediction_ui(client: ESM3InferenceClient) -> widgets.Widget:
    # Alow a single protein and immediately load it from workspace
    protein_importer = ProteinImporter(max_proteins=1, autoload=True)

    sequence_input_ui = widgets.VBox(
        [
            widgets.HTML(value="<h3>Or enter a protein sequence:</h3>"),
            widgets.Textarea(
                placeholder="Enter protein sequence",
                layout=widgets.Layout(width="400px", height="100px"),
            ),
        ]
    )

    input_ui = widgets.Tab(children=[sequence_input_ui, protein_importer.importer_ui])
    input_ui.set_title(0, "Enter Sequence")
    input_ui.set_title(1, "Add Protein")

    predict_button = widgets.Button(
        description="Predict",
        disabled=True,
        tooltip="Click to predict the protein's properties",
        style={"button_color": "lightgreen"},
    )

    output = widgets.Output()

    prediction_ui = widgets.VBox([input_ui, output])

    def get_protein() -> ESMProtein:
        if input_ui.selected_index == 1:
            [first_protein] = protein_importer.protein_list
            protein_id, protein_chain = first_protein
            protein = ESMProtein.from_protein_chain(protein_chain)

            # NOTE: We ignore all properties except sequence and structure
            protein.secondary_structure = None
            protein.sasa = None
            protein.function_annotations = None
            return protein
        else:
            sequence = sequence_input_ui.children[1].value
            return ESMProtein(sequence=sequence)

    def on_new_sequence(_):
        is_sequence = len(sequence_input_ui.children[1].value) > 0
        predict_button.disabled = not is_sequence
        prediction_ui.children = [input_ui, predict_button, output]

    def on_new_protein(_):
        is_protein = len(protein_importer.protein_list) > 0
        predict_button.disabled = not is_protein
        prediction_ui.children = [input_ui, predict_button, output]

    def validate_predict(_):
        if input_ui.selected_index == 1:
            if len(protein_importer.protein_list) > 0:
                predict_button.disabled = False
            else:
                predict_button.disabled = True
        else:
            if len(sequence_input_ui.children[1].value) == 0:
                predict_button.disabled = True
            else:
                predict_button.disabled = False

    def on_click_predict(_):
        try:
            # Reset the output and results
            output.clear_output()
            prediction_ui.children = [input_ui, predict_button, output]
            # Predict the protein's properties
            with output:
                protein = get_protein()

                tracks = ["structure", "secondary_structure", "sasa", "function"]

                success = True
                for track in tracks:
                    print(f"Predicting {track}...")
                    protein = client.generate(
                        protein, config=GenerationConfig(track=track)
                    )
                    if isinstance(protein, ESMProteinError):
                        wrapped_print(f"Protein Error: {protein.error_msg}")
                        success = False

                assert isinstance(protein, ESMProtein)

                if success:
                    structure_results = create_results_visualizer(
                        modality="structure",
                        samples=[protein],
                        items_per_page=1,
                        include_title=False,
                    )
                    secondary_structure_results = create_results_visualizer(
                        modality="secondary_structure",
                        samples=[protein],
                        items_per_page=1,
                        include_title=False,
                    )
                    sasa_results = create_results_visualizer(
                        modality="sasa",
                        samples=[protein],
                        items_per_page=1,
                        include_title=False,
                    )
                    function_results = create_results_visualizer(
                        modality="function",
                        samples=[protein],
                        items_per_page=1,
                        include_title=False,
                    )
                    results_ui = widgets.Tab(
                        children=[
                            structure_results,
                            secondary_structure_results,
                            sasa_results,
                            function_results,
                        ]
                    )
                    results_ui.set_title(0, "Structure")
                    results_ui.set_title(1, "Secondary Structure")
                    results_ui.set_title(2, "SASA")
                    results_ui.set_title(3, "Function")

                    output.clear_output()
                    prediction_ui.children = [
                        input_ui,
                        predict_button,
                        output,
                        results_ui,
                    ]

        except Exception as e:
            with output:
                wrapped_print(e)

    predict_button.on_click(on_click_predict)
    protein_importer.entries_box.observe(on_new_protein, names="children")
    protein_importer.register_delete_callback(lambda: validate_predict(None))

    sequence_input_ui.children[1].observe(on_new_sequence, names="value")
    input_ui.observe(validate_predict, names="selected_index")

    return prediction_ui
