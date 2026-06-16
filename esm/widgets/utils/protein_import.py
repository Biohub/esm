import codecs
from io import StringIO
from typing import Callable

from ipywidgets import widgets

from esm.utils.structure.protein_chain import ProteinChain
from esm.widgets.utils.printing import wrapped_print


class ProteinImporter:
    def __init__(self, max_proteins: int | None = None, autoload: bool = False) -> None:
        self._protein_list: list[tuple[str, ProteinChain]] = []
        self._protein_workspace: dict[str, str] = {}
        self.max_proteins = max_proteins
        self.autoload = autoload

        # Workspace section
        self.workspace_title = widgets.HTML(
            value="<b>Workspace:</b>", layout=widgets.Layout(margin="0 0 10px 0")
        )
        self.workspace = widgets.VBox(children=[])  # Start with empty workspace
        self.pdb_uploader = widgets.FileUpload(
            description="Upload PDB file to workspace",
            accept=".pdb",
            layout=widgets.Layout(width="310px"),
        )
        self.workspace_section = widgets.VBox(
            [self.workspace_title, self.workspace, self.pdb_uploader],
            layout=widgets.Layout(gap="10px"),
        )

        # Add protein section
        self.add_protein_section_title = widgets.HTML(
            value="<b>Add Reference Proteins from the Workspace or RCSB:</b>",
            layout=widgets.Layout(width="400px"),
        )
        self.pdb_id_input = widgets.Text(
            description="PDB ID:",
            placeholder="Enter PDB ID or Filename",
            layout=widgets.Layout(width="400px"),
        )
        self.pdb_chain_input = widgets.Text(
            description="Chain:",
            placeholder="Enter chain ID",
            layout=widgets.Layout(width="400px"),
        )
        self.pdb_id_add_button = widgets.Button(
            description="Add", layout=widgets.Layout(width="100px")
        )
        self.add_protein_section = widgets.VBox(
            [
                self.add_protein_section_title,
                self.pdb_id_input,
                self.pdb_chain_input,
                self.pdb_id_add_button,
            ]
        )

        self.error_output = widgets.Output()
        self.entries_box = widgets.VBox()

        self.pdb_id_add_button.on_click(self.on_click_add)
        self.pdb_uploader.observe(self.on_upload, names="value")

        self.delete_callbacks: list[Callable[[], None]] = []

        self.importer_ui = widgets.VBox(
            [
                self.workspace_section,
                self.add_protein_section,
                self.error_output,
                self.entries_box,
            ],
            layout=widgets.Layout(gap="10px", width="600px"),
        )

    @property
    def protein_list(self):
        return self._protein_list

    def on_click_add(self, _):
        pdb_id = self.pdb_id_input.value
        chain_id = self.pdb_chain_input.value or "detect"
        self.add_pdb_id(pdb_id, chain_id)

    def add_pdb_id(self, pdb_id: str, chain_id: str):
        try:
            self.error_output.clear_output()

            if self.max_proteins and len(self._protein_list) >= self.max_proteins:
                raise ValueError("Maximum number of proteins reached")

            if not pdb_id:
                raise ValueError("PDB ID or Filename is required")
            if pdb_id.lower().endswith(".pdb"):
                try:
                    str_content = self._protein_workspace[pdb_id]
                    protein = ProteinChain.from_pdb(
                        StringIO(str_content), chain_id=chain_id
                    )
                    chain_id = protein.chain_id
                except KeyError:
                    raise ValueError("PDB file not found in workspace")
            else:
                protein = ProteinChain.from_rcsb(pdb_id=pdb_id, chain_id=chain_id)
                chain_id = protein.chain_id
            self._protein_list.append((f"{pdb_id} Chain:{chain_id}", protein))
            self.add_entry_to_ui(f"{pdb_id} Chain:{chain_id}")
        except Exception as e:
            with self.error_output:
                wrapped_print(f"Error: {e}")

    def add_entry_to_ui(self, protein_id: str):
        entry_button = widgets.Button(description="Remove")
        entry_label = widgets.Label(value=protein_id)
        entry_label.tag = protein_id
        entry_container = widgets.HBox([entry_button, entry_label])

        def delete_entry(b):
            self.entries_box.children = [
                child for child in self.entries_box.children if child != entry_container
            ]
            self._protein_list = [
                protein for protein in self._protein_list if protein[0] != protein_id
            ]
            for callback in self.delete_callbacks:
                callback()

        entry_button.on_click(delete_entry)
        self.entries_box.children += (entry_container,)

    def on_upload(self, _):
        try:
            self.error_output.clear_output()

            if self.max_proteins and len(self._protein_list) >= self.max_proteins:
                raise ValueError("Maximum number of proteins reached")

            uploaded_file = next(iter(self.pdb_uploader.value))
            filename: str = uploaded_file["name"]
            str_content = codecs.decode(uploaded_file["content"], encoding="utf-8")
            self._protein_workspace[filename] = str_content
            self.workspace.children += (widgets.Label(value=f"{filename}"),)

            if self.autoload:
                self.add_pdb_id(filename, "detect")

        except Exception as e:
            with self.error_output:
                wrapped_print(f"Error: {e}")

    def toggle_no_protein(self, change):
        if change["new"]:
            self.pdb_id_input.disabled = True
            self.pdb_chain_input.disabled = True
            self.pdb_id_add_button.disabled = True
            self.pdb_uploader.disabled = True
        else:
            self.pdb_id_input.disabled = False
            self.pdb_chain_input.disabled = False
            self.pdb_id_add_button.disabled = False
            self.pdb_uploader.disabled = False

    def register_delete_callback(self, callback: Callable[[], None]):
        self.delete_callbacks.append(callback)
