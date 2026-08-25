from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput


def validate_fold_max_accuracy_input(all_atom_input: StructurePredictionInput) -> None:
    """Reject inputs ``fold_max_accuracy`` cannot serve.

    Parameters
    ----------
    all_atom_input
        The molecular complex to fold.

    Raises
    ------
    ValueError
        If the complex is empty, or if any protein chain carries an MSA.
    """
    if len(all_atom_input.sequences) == 0:
        raise ValueError("Input sequence length is 0. Please provide a valid input.")
    for seq in all_atom_input.sequences:
        if isinstance(seq, ProteinInput) and seq.msa is not None:
            raise ValueError(
                "fold_max_accuracy generates its own MSA and cannot fold against a supplied one."
            )
