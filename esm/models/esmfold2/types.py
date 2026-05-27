"""Re-exports of the canonical SPI dataclasses from input_builder.

This module exists so the HF processor and downstream code can import the
ESMFold2 input types from a single namespace without picking up internal-only
sibling utilities. The actual definitions live in
``esm.utils.structure.input_builder``.
"""

from esm.utils.msa import MSA
from esm.utils.parsing import FastaEntry
from esm.utils.structure.input_builder import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

__all__ = [
    "FastaEntry",
    "MSA",
    "Modification",
    "ProteinInput",
    "RNAInput",
    "DNAInput",
    "LigandInput",
    "DistogramConditioning",
    "CovalentBond",
    "StructurePredictionInput",
]
