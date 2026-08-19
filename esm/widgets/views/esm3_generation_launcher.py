import datetime
import traceback
from typing import Any, Callable, Literal

from ipywidgets import widgets

from esm.models.esm3 import ESM3
from esm.sdk import ESM3ForgeInferenceClient
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    GenerationConfig,
)
from esm.utils.constants import models
from esm.widgets.components.results_visualizer import (
    create_results_visualizer,
)
from esm.widgets.utils.printing import wrapped_print
from esm.widgets.utils.serialization import (
    create_download_results_button,
)


def create_esm3_generation_launcher(
    protein: ESMProtein,
    client: ESM3InferenceClient | None = None,
    forge_token: str = "",
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
) -> widgets.Widget:
    if isinstance(client, ESM3):
        model_name_ = models.ESM3_OPEN_SMALL
    elif isinstance(client, ESM3ForgeInferenceClient):
        model_name_ = client.model
    else:
        model_name_ = models.ESM3_OPEN_SMALL.replace("_", "-")

    model_name = widgets.Text(
        description="Model: ",
        value=model_name_,
        disabled=True if client is not None else False,
    )

    track = widgets.Dropdown(
        options=["sequence", "structure", "secondary_structure", "sasa", "function"],
        description="Track:",
        disabled=False,
    )
    num_steps = widgets.IntSlider(
        value=1,
        min=1,
        max=len(protein),
        step=1,
        description="Num Steps:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )
    temperature = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="Temperature:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )
    top_p = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Top P:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".2f",
    )
    num_samples = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        step=1,
        description="Num Samples:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )
    output = widgets.Output()

    generate_button = widgets.Button(
        description="Generate",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click to generate proteins with ESM-3",
    )

    generation_config_settings_ui = widgets.VBox(
        [
            model_name,
            track,
            num_steps,
            temperature,
            top_p,
            num_samples,
            generate_button,
            output,
        ]
    )

    generation_config_ui = widgets.VBox([generation_config_settings_ui])

    def on_track_change(change):
        if change["new"] == "function":
            num_steps.value = 1
            num_steps.max = 1
        else:
            num_steps.max = len(protein)

    def on_generate(*args, **kwargs):
        if not track.value:
            with output:
                print("Please select a track.")
            return

        config = GenerationConfig(
            track=track.value,
            num_steps=num_steps.value,
            temperature=temperature.value,
            top_p=top_p.value,
        )
        with output:
            output.clear_output()
            print(f"Generating {num_samples.value} samples...")
            try:
                if client is None:
                    client_ = ESM3ForgeInferenceClient(
                        model=model_name.value, token=forge_token
                    )
                elif isinstance(client, ESM3):
                    if (
                        models.normalize_model_name(model_name.value)
                        != models.ESM3_OPEN_SMALL
                    ):
                        raise ValueError(
                            f"Model name {model_name.value} does not match the client model {models.ESM3_OPEN_SMALL}"
                        )
                    client_ = client
                elif isinstance(client, ESM3ForgeInferenceClient):
                    if model_name.value != client.model:
                        raise ValueError(
                            f"Model name {model_name.value} does not match the client model {client.model}"
                        )
                    client_ = client
                else:
                    raise ValueError("Invalid client type")

                proteins: list[ESMProtein] = client_.batch_generate(
                    [protein] * num_samples.value, configs=[config] * num_samples.value
                )  # type: ignore

                is_error = False
                for out_protein in proteins:
                    if isinstance(out_protein, ESMProteinError):
                        wrapped_print(f"Protein Error: {out_protein.error_msg}")
                        is_error = True

                if not is_error:
                    print(f"Generated {len(proteins)} proteins.")
                else:
                    return
            except Exception:
                # Add protein information to error message
                tb_str = traceback.format_exc()
                error_message = (
                    f"An error occurred:\n{tb_str}\n\n"
                    "Protein information:\n"
                    f"Sequence: {protein.sequence}\n"
                    f"Coordinates (Structure): {protein.coordinates}\n"
                    f"Secondary Structure: {protein.secondary_structure}\n"
                    f"SASA: {protein.sasa}\n"
                    f"Function: {protein.function_annotations}\n\n"
                    "Config information:\n"
                    f"Model: {model_name.value}\n"
                    f"Track: {track.value}\n"
                    f"Num Steps: {num_steps.value}\n"
                    f"Temperature: {temperature.value}\n"
                    f"Top P: {top_p.value}\n"
                    f"Num Samples: {num_samples.value}\n"
                )
                wrapped_print(error_message)

        results_visualizer = create_results_visualizer(
            track.value, proteins, copy_to_prompt_callback=copy_to_prompt_callback
        )
        generation_config_ui.children = [
            generation_config_settings_ui,
            results_visualizer,
        ]

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"generated_proteins_{track.value}_{timestamp}.json"
        download_button = create_download_results_button(proteins, filename)
        generation_config_ui.children = [
            *generation_config_ui.children,
            download_button,
        ]

    generate_button.on_click(on_generate)
    track.observe(on_track_change, names="value")

    return generation_config_ui
