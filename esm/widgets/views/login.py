import os
from functools import partial
from textwrap import dedent

from ipywidgets import widgets

from esm.widgets.utils.clients import get_forge_client, get_local_client
from esm.widgets.utils.types import ClientInitContainer


def create_login_ui(client_container: ClientInitContainer):
    # ======== Info ========
    info_content = """
    <div style="font-family: Arial, sans-serif;">
        <h2>Inference Options for Our Model</h2>

        <p><strong>Forge/Biohub Platform API (remote):</strong> Our hosted service that provides access to the full suite of ESM3 models.
        To utilize the Forge/Biohub Platform API, users must first agree to the
        <a href="https://biohub.org/terms-of-use/" target="_blank">Terms of Service</a>
        and generate an access token via the
        <a href="https://biohub.ai/" target="_blank">Biohub Platform</a>.
        The console also provides a comprehensive list of models available to each user.</p>
        <p><strong>Local:</strong> Self-hosted solution that supports the <code>esm3-sm-open-v1</code> model exclusively.
        A GPU-enabled instance is recommended for running the model locally. All data processing with the Local API is conducted on the user's machine, ensuring that no data is transmitted to EvolutionaryScale servers.</p>
    </div>
    """

    infobox = widgets.HTML(value=info_content)

    selection_ui = widgets.VBox(
        [
            widgets.HTML(value="<h3>Select an Inference Option</h3>"),
            widgets.RadioButtons(
                options=["Forge/Biohub Platform API", "Local"],
                value=None,
                description="",
                disabled=False,
            ),
        ]
    )

    start_button = widgets.Button(
        description="Start using ESM3", layout=widgets.Layout(margin="50px 0 0 0")
    )
    start_msg_output = widgets.Output()

    # ======== Login ========

    login_ui = widgets.VBox([infobox, selection_ui])

    forge = widgets.VBox()

    # If not logged in, show login form
    forge_info = widgets.HTML(
        value="<p>Copy a token from your <a href='https://biohub.ai/console'>Forge console page</a> and paste it below:</p>"
    )
    forge_token_input = widgets.Text(
        description="Token:",
        placeholder="Paste your token here",
        layout={"width": "50%"},
    )
    forge_set_as_env = widgets.Checkbox(
        value=True, description="Set as environment variable"
    )
    forge_login = widgets.Button(description="Login")
    forge_login_view = widgets.VBox(
        [forge_info, forge_token_input, forge_set_as_env, forge_login]
    )

    # If logged in, show info view
    forge_logged_in = widgets.HTML(
        value="<p>You are already logged in. You can now use the Forge/Biohub Platform API.</p>"
    )
    forge_login_with_new_token = widgets.Button(description="Use different token")
    forge_logged_in_view = widgets.VBox([forge_logged_in, forge_login_with_new_token])

    forge_token = os.environ.get("ESM_API_KEY", None)
    if forge_token:
        forge.children = [forge_logged_in_view]
    else:
        forge.children = [forge_login_view]

    # ======== Model Selection ========

    local_model = widgets.Text(
        value="esm3-sm-open-v1",
        description="Model Name:",
        disabled=True,
        layout={"width": "50%"},
    )
    forge_model = widgets.Text(
        value="esm3-open",
        description="Model Name:",
        disabled=False,
        layout={"width": "50%"},
    )

    forge_model_selection_info = widgets.HTML(
        value="<p>Enter the model name from the <a href='https://biohub.ai/console'>Forge console page</a> that you would like to use:</p>"
    )

    model_selection_header = widgets.HTML(value="<h3>Select a Model</h3>")
    model_selection_ui = widgets.VBox([model_selection_header])

    # ======== Callbacks ========

    def on_forge_login_clicked(b):
        forge_token = forge_token_input.value
        if forge_set_as_env.value:
            os.environ["ESM_API_KEY"] = forge_token
            forge.children = [forge_logged_in_view]

    def on_forge_login_with_new_token_clicked(b):
        forge.children = [forge_login_view]

    def on_selection_change(change):
        client_container.metadata["inference_option"] = change["new"]
        if change["new"] == "Forge/Biohub Platform API":
            model_selection_ui.children = [
                model_selection_header,
                forge_model_selection_info,
                forge_model,
            ]
            login_ui.children = [
                infobox,
                selection_ui,
                forge,
                model_selection_ui,
                start_button,
                start_msg_output,
            ]
        elif change["new"] == "Local":
            model_selection_ui.children = [model_selection_header, local_model]
            login_ui.children = [
                infobox,
                selection_ui,
                model_selection_ui,
                start_button,
                start_msg_output,
            ]

    def on_start(*args):
        if selection_ui.children[1].value == "Forge/Biohub Platform API":
            client_container.client_init_callback = partial(
                get_forge_client, forge_model.value
            )
        else:
            client_container.client_init_callback = partial(get_local_client)

        start_msg_output.clear_output()
        with start_msg_output:
            msg = dedent(
                f"""
                Parameters for the ESM3 Inference Client:
                - Inference Option: {selection_ui.children[1].value}
                - Model Name: {forge_model.value if selection_ui.children[1].value == "Forge/Biohub Platform API" else local_model.value}

                Please initialize the client by executing the next code cell in the notebook.
                """
            )
            print(msg)

    forge_login.on_click(on_forge_login_clicked)
    forge_login_with_new_token.on_click(on_forge_login_with_new_token_clicked)

    selection_ui.children[1].observe(on_selection_change, names="value")

    start_button.on_click(on_start)

    return login_ui
