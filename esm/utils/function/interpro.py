"""Utilities for interacting with InterPro."""

import itertools
import re
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cached_property

import networkx as nx
import pandas as pd
from cloudpathlib import AnyPath

from esm.utils.constants import esm3 as C
from esm.utils.types import PathLike


def parse_go_terms(text: str) -> list[str]:
    """Parses GO terms from a string.

    Args:
        text: String containing GO terms. Example: "GO:0008309, GO:1902267" Note that GO
          terms have exactly 7 digits.
    Returns:
        All GO terms found in the string. Example: ['GO:0008309', 'GO:1902267']
    """
    return re.findall(r"GO:(?:\d{7,})", text)


def _parse_interpro2go(path: PathLike) -> dict[str, list[str]]:
    """Parses InterPro2GO file into map.

    NOTE: this file has a very strange, non-standard format.

    Args:
        path: path to InterPro2GO file from: https://www.ebi.ac.uk/GOA/InterPro2GO
    Returns:
        Mapping from InterPro to list of associated GO terms.
    """
    with AnyPath(path).open("r") as f:
        text = f.read()
    df = pd.Series(text.split("\n"), name="line").to_frame()
    df = df[~df.line.str.startswith("!")]
    df["interpro_id"] = df.line.apply(lambda line: re.findall(r"IPR\d+", line))
    df["go_ids"] = df.line.apply(parse_go_terms)
    df = df[df.go_ids.apply(len).gt(0) & df.interpro_id.apply(len).eq(1)]
    df["interpro_id"] = df["interpro_id"].apply(lambda xs: xs[0])

    # Group all mappints together into a single map.
    df = (
        df.groupby("interpro_id")["go_ids"]
        .apply(lambda group: list(itertools.chain.from_iterable(group)))
        .reset_index()
    )
    return dict(zip(df.interpro_id, df.go_ids))


class InterProEntryType(IntEnum):
    """InterPro types and representation counts:

    Family                    21,942
    Domain                    14,053
    Homologous_superfamily     3,446
    Conserved_site               728
    Repeat                       374
    Active_site                  133
    Binding_site                  75
    PTM                           17
    """

    ACTIVE_SITE = 0
    BINDING_SITE = auto()
    CONSERVED_SITE = auto()
    DOMAIN = auto()
    FAMILY = auto()
    HOMOLOGOUS_SUPERFAMILY = auto()
    PTM = auto()
    REPEAT = auto()
    UNKNOWN = auto()


@dataclass
class InterProEntry:
    """Represents an InterPro entry."""

    id: str  # Example: IPR000006
    type: InterProEntryType
    name: str  # Example: "Metallothionein, vertebrate"
    description: str | None = None


class InterPro:
    """Convenience class interacting with InterPro ontology/data."""

    def __init__(
        self,
        entries_path: PathLike | None = None,
        hierarchy_path: PathLike | None = None,
        interpro2go_path: PathLike | None = None,
    ):
        """Constructs interface to query InterPro entries."""

        def default(x, d):
            return x if x is not None else d

        self.entries_path = default(entries_path, C.INTERPRO_ENTRY)
        self.hierarchy_graph_path = default(hierarchy_path, C.INTERPRO_HIERARCHY)
        self.interpro2go_path = default(interpro2go_path, C.INTERPRO2GO)

    @cached_property
    def interpro2go(self) -> dict[str, list[str]]:
        """Reads the InterPro to GO term mapping."""
        assert self.interpro2go_path is not None
        return _parse_interpro2go(self.interpro2go_path)

    @cached_property
    def entries_frame(self) -> pd.DataFrame:
        """Loads full InterPro entry set as a DataFrame.

        Colums are
            - "id": str interpro accession /id as
            - "type": InterProEntryType representing the type of annotation.
            - "name": Short name of the entry.
        """
        with AnyPath(self.entries_path).open("r") as f:
            df = pd.read_csv(f, sep="\t")
        assert all(
            col in df.columns for col in ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"]
        )
        df.rename(
            columns={"ENTRY_AC": "id", "ENTRY_TYPE": "type", "ENTRY_NAME": "name"},
            inplace=True,
        )
        df["type"] = df.type.str.upper().apply(
            lambda type_name: InterProEntryType[type_name]
        )
        return df

    @cached_property
    def entries(self) -> dict[str, InterProEntry]:
        """Returns all InterPro entries."""
        return {
            row.id: InterProEntry(  # type: ignore
                id=row.id,  # type: ignore
                type=row.type,  # type: ignore
                name=row.name,  # type: ignore
            )
            for row in self.entries_frame.itertuples()
        }

    def lookup_name(self, interpro_id: str) -> str | None:
        """Short name / title for an interpro id."""
        if interpro_id not in self.entries:
            return None
        return self.entries[interpro_id].name

    def lookup_entry_type(self, interpro_id: str) -> InterProEntryType:
        """Looks up entry-type for an interpro id."""
        if interpro_id in self.entries:
            return self.entries[interpro_id].type
        else:
            return InterProEntryType.UNKNOWN

    @cached_property
    def graph(self) -> nx.DiGraph:
        """Reads the InterPro hierarchy of InterPro."""
        graph = nx.DiGraph()
        with AnyPath(self.hierarchy_graph_path).open("r") as f:
            parents = []
            for line in f:
                ipr = line.split("::", maxsplit=1)[0]
                ipr_strip = ipr.lstrip("-")
                level = (len(ipr) - len(ipr_strip)) // 2
                parents = parents[:level]
                graph.add_node(ipr_strip)
                if parents:
                    graph.add_edge(ipr_strip, parents[-1])
                parents.append(ipr_strip)
        return graph
