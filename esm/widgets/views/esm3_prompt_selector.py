from ipywidgets import widgets

from esm.widgets.components.sasa_prompt_selector import (
    create_sasa_prompt_selector,
)
from esm.widgets.components.secondary_structure_prompt_selector import (
    create_secondary_structure_prompt_selector,
)
from esm.widgets.components.sequence_prompt_selector import (
    create_sequence_prompt_selector,
)
from esm.widgets.components.structure_prompt_selector import (
    create_structure_prompt_selector,
)
from esm.widgets.utils.prompting import PromptManagerCollection
from esm.widgets.utils.protein_import import ProteinImporter


def create_esm3_prompt_selector(
    prompt_manager_collection: PromptManagerCollection,
    protein_importer: ProteinImporter,
):
    # Placeholder for the accordion and tabs
    sequence_tabs = widgets.Tab()
    structure_tabs = widgets.Tab()
    secondary_structure_tabs = widgets.Tab()
    sasa_tabs = widgets.Tab()

    accordion = widgets.Accordion(
        children=[sequence_tabs, structure_tabs, secondary_structure_tabs, sasa_tabs]
    )
    accordion.set_title(0, "Sequence")
    accordion.set_title(1, "Structure")
    accordion.set_title(2, "Secondary Structure")
    accordion.set_title(3, "Solvent Accessible Surface Area (SASA)")

    def create_active_callback(tabs):
        def get_active_tag():
            selected_tab_index = tabs.selected_index
            title = tabs.get_title(selected_tab_index)
            return title

        return get_active_tag

    sequence_children = []
    structure_children = []
    secondary_structure_children = []
    sasa_children = []

    prompt_manager_collection.reset_all_handlers()

    for i, (protein_id, protein_chain) in enumerate(protein_importer.protein_list):
        sequence_prompt_selector = create_sequence_prompt_selector(
            prompt_manager_collection.sequence_prompt_manager,
            tag=protein_id,
            full_sequence=protein_chain.sequence,
            with_title=False,
            active_tag_callback=create_active_callback(sequence_tabs),
        )
        structure_prompt_selector = create_structure_prompt_selector(
            prompt_manager_collection.structure_prompt_manager,
            tag=protein_id,
            protein_chain=protein_chain,
            with_title=False,
            active_tag_callback=create_active_callback(structure_tabs),
        )
        secondary_structure_prompt_selector = (
            create_secondary_structure_prompt_selector(
                prompt_manager_collection.secondary_structure_prompt_manager,
                tag=protein_id,
                protein_chain=protein_chain,
                with_title=False,
                active_tag_callback=create_active_callback(secondary_structure_tabs),
            )
        )
        sasa_prompt_selector = create_sasa_prompt_selector(
            prompt_manager_collection.sasa_prompt_manager,
            tag=protein_id,
            protein_chain=protein_chain,
            with_title=False,
            active_tag_callback=create_active_callback(sasa_tabs),
        )

        sequence_children.append(sequence_prompt_selector)
        structure_children.append(structure_prompt_selector)
        secondary_structure_children.append(secondary_structure_prompt_selector)
        sasa_children.append(sasa_prompt_selector)

    sequence_tabs.children = sequence_children
    structure_tabs.children = structure_children
    secondary_structure_tabs.children = secondary_structure_children
    sasa_tabs.children = sasa_children

    for i, (protein_id, _) in enumerate(protein_importer.protein_list):
        sequence_tabs.set_title(i, protein_id)
        structure_tabs.set_title(i, protein_id)
        secondary_structure_tabs.set_title(i, protein_id)
        sasa_tabs.set_title(i, protein_id)

    return accordion
