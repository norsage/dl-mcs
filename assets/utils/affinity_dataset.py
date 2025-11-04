import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gemmi
import pandas as pd
import plotly.graph_objects as go
import torch
from jaxtyping import Bool, Float32, Int
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import remove_isolated_nodes, to_undirected


class InterfaceGraph(Protocol):
    atoms: Int[Tensor, "n"]
    residues: Int[Tensor, "n"]
    coordinates: Float32[Tensor, "n 3"]
    receptor_mask: Bool[Tensor, "n"]
    edge_index: Int[Tensor, "2 e"]
    distances: Float32[Tensor, "e"]
    batch: Int[Tensor, "n"]
    y: Float32[Tensor, "n"]


def read_structure(pdb: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(pdb))
    # setup entities, but do not remove duplicate chains
    # structure.setup_entities()
    structure.add_entity_types()
    structure.assign_subchains()
    structure.ensure_entities()
    # clean
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()
    structure.cell = gemmi.UnitCell()  # for correct contacts search
    return structure


def find_contacts(
    structure: gemmi.Structure,
    receptor_chains: list[str],
    ligand_chains: list[str],
    radius: float,
) -> list[gemmi.ContactSearch.Result]:
    cs = gemmi.ContactSearch(radius)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain
    cs.twice = True

    # keep only necessary chains
    sel = gemmi.Selection(",".join(receptor_chains + ligand_chains))
    sel.remove_not_selected(structure)

    ns = gemmi.NeighborSearch(structure, radius).populate()
    contacts = cs.find_contacts(ns)
    contacts = [x for x in contacts if is_outer_contact(x, receptor_chains)]
    return contacts


def is_outer_contact(
    contact: gemmi.ContactSearch.Result, receptor_chains: list[str]
) -> bool:
    src_is_receptor = contact.partner1.chain.name in receptor_chains
    dst_is_receptor = contact.partner2.chain.name in receptor_chains
    return src_is_receptor ^ dst_is_receptor


# fmt: off
ATOM_NAMES = [
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2", "CH2",
    "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT",
    "SD", "SG",
]
ATOM_COLORS = {
    "C": "gray",
    "N": "blue",
    "O": "red",
    "H": "black",
    "S": "yellow",
}
ATOMS_INDICES = {x: i for i, x in enumerate(ATOM_NAMES, start=1)}
RESIDUES = "ARNDCQEGHILKMFPSTWYV"
RESIDUE_INDICES = {x: i for i, x in enumerate(RESIDUES, start=1)}
# fmt: on


@dataclass(frozen=True)
class DataItem:
    uid: str
    pdb: Path
    receptor_chains: list[str]
    ligand_chains: list[str]
    dG: float


class InterfaceGraphBuilder:
    def build_graph(self, item: DataItem) -> InterfaceGraph: ...


class AtomicInterfaceGraphBuilder:
    def __init__(
        self,
        interface_distance: float,
        radius: float,
        keep_inner_edges: bool = True,
        max_neighbors: int = 30,
    ):
        super().__init__()
        self.interface_distance = interface_distance
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.keep_inner_edges = keep_inner_edges

    def build_graph(self, item: DataItem) -> InterfaceGraph:
        # prepare contacts
        structure = read_structure(item.pdb)
        contacts = find_contacts(
            structure, item.receptor_chains, item.ligand_chains, self.interface_distance
        )
        # create mapping from atom to index
        atom_to_id: dict[tuple[gemmi.Atom, str, str], int] = {}
        for contact in contacts:
            if (contact.partner1.atom, contact.partner1.chain.name) not in atom_to_id:
                key = (
                    contact.partner1.atom,
                    contact.partner1.residue.name,
                    contact.partner1.chain.name,
                )
                atom_to_id[key] = len(atom_to_id)
        atom_to_id = atom_to_id

        # build graph
        atom_indices = torch.tensor(
            [ATOMS_INDICES.get(atom.name, 0) for atom, _, _ in atom_to_id.keys()]
        )
        residue_names = gemmi.one_letter_code(
            [residue_name for _, residue_name, _ in atom_to_id.keys()]
        )
        residue_indices = torch.tensor(
            [RESIDUE_INDICES.get(x, 0) for x in residue_names]
        )
        coordinates = torch.tensor(
            [x.pos.tolist() for x, _, _ in atom_to_id.keys()], dtype=torch.float32
        )
        receptor_mask = torch.tensor(
            [
                int(chain_id in item.receptor_chains)
                for _, _, chain_id in atom_to_id.keys()
            ]
        )

        edge_index = to_undirected(
            radius_graph(
                coordinates, max_num_neighbors=self.max_neighbors, r=self.radius
            )
        )
        if not self.keep_inner_edges:
            src, dst = edge_index
            is_intermol = receptor_mask[src] != receptor_mask[dst]
            edge_index = edge_index[:, is_intermol]

        # distances between atoms
        src, dst = edge_index
        distances = (coordinates[src] - coordinates[dst]).norm(dim=1)

        return Data(
            atoms=atom_indices,
            residues=residue_indices,
            coordinates=coordinates,
            receptor_mask=receptor_mask,
            edge_index=edge_index,
            distances=distances,
            num_nodes=len(atom_indices),
            y=item.dG,
        )


class AffinityDataset:
    def __init__(
        self,
        datadir: Path,
        subset_csv: Path,
        graph_builder: InterfaceGraphBuilder,
    ) -> None:
        super().__init__()
        self.graph_builder = graph_builder
        self.data = self._preprocess(subset_csv, datadir)

    def _rows_to_items(self, subset_df: pd.DataFrame, datadir: Path) -> list[DataItem]:
        items = []
        for _, row in subset_df.iterrows():
            uid = row["uid"].lower()
            pdb_path = datadir / f"{uid}.pdb"
            if not pdb_path.is_file():
                continue
            item = DataItem(
                uid, pdb_path, row["receptor_chains"], row["ligand_chains"], row["dG"]
            )
            items.append(item)
        return items

    def _preprocess_one(self, item: DataItem) -> InterfaceGraph:
        return self.graph_builder.build_graph(item)

    def _preprocess(self, subset_csv: Path, datadir: Path) -> list[InterfaceGraph]:
        subset_df = pd.read_csv(subset_csv)
        items = self._rows_to_items(subset_df, datadir)
        data = list(map(self._preprocess_one, items))
        return data

    def __getitem__(self, index: int) -> tuple[InterfaceGraph, DataItem]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class PlotlyVis:
    @classmethod
    def create_figure(
        cls,
        graph: InterfaceGraph,
        receptor_color: str = "teal",
        ligand_color: str = "coral",
        figsize: tuple[int, int] = (900, 600),
    ) -> go.Figure:
        traces = cls.plot_graph(graph, receptor_color, ligand_color)
        width, height = figsize
        figure = go.Figure(
            data=traces,
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                showlegend=False,
                width=width,
                height=height,
                # plot_bgcolor="rgba(0, 0, 0, 1)",
                # paper_bgcolor="rgba(0, 0, 0, 1)",
            ),
        )
        return figure

    @classmethod
    def plot_graph(
        cls,
        graph: InterfaceGraph,
        receptor_color: str,
        ligand_color: str,
    ) -> go.Figure:
        assert graph.coordinates is not None
        assert graph.edge_index is not None
        # получим ковалентные связи, чтобы нарисовать их по-другому
        ligand_cov, receptor_cov = cls.get_covalent_edges_masks(
            graph, distance_threshold=2.0
        )
        # нарисуем
        data = [
            # вершины рецептора
            cls.draw_nodes(graph, graph.receptor_mask == 1),
            # вершины лиганда
            cls.draw_nodes(graph, graph.receptor_mask == 0),
            # ковалентные связи лиганда
            cls.draw_edges(
                graph,
                edges_mask=ligand_cov,
                add_annotation=False,
                color=ligand_color,
                dash="solid",
                width=5,
            ),
            # ковалентные связи рецептора
            cls.draw_edges(
                graph,
                edges_mask=receptor_cov,
                add_annotation=False,
                color=receptor_color,
                dash="solid",
                width=5,
            ),
            # все связи в графе
            cls.draw_edges(
                graph,
                edges_mask=None,
                add_annotation=True,
                color="lightgray",
                dash="dot",
                width=1,
            ),
        ]
        return data

    @staticmethod
    def get_covalent_edges_masks(
        graph: InterfaceGraph, distance_threshold: float = 2.2
    ) -> list[Tensor]:
        src, tgt = graph.edge_index
        covalent_masks = []
        for chain_id in graph.receptor_mask.unique():  # type: ignore[no-untyped-call]
            chain_atoms = graph.receptor_mask == chain_id
            chain_edges = (
                chain_atoms[src]
                * chain_atoms[tgt]
                * (graph.distances <= distance_threshold)
            )
            covalent_masks.append(chain_edges)
        return covalent_masks

    @staticmethod
    def draw_nodes(
        graph: InterfaceGraph, nodes_mask: Tensor | None = None
    ) -> go.Scatter3d:
        x, y, z = graph.coordinates[nodes_mask].T
        atom_types = [ATOM_NAMES[x.item() - 1][0] for x in graph.atoms[nodes_mask]]
        atom_colors = [ATOM_COLORS[x] for x in atom_types]
        nodes = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            hoverinfo="text",
            text=[ATOM_NAMES[x.item() - 1] for x in graph.atoms[nodes_mask]],
            marker=dict(
                size=4,
                color=atom_colors,
                cmin=0,
                cmax=1,
                opacity=0.8,
            ),
        )
        return nodes

    @staticmethod
    def draw_edges(
        graph: InterfaceGraph,
        edges_mask: Tensor | None = None,
        add_annotation: bool = False,
        color: str = "lightgray",
        dash: str = "dot",
        width: int = 1,
    ) -> go.Scatter3d:
        selected_edges, distances = graph.edge_index.T, graph.distances
        if edges_mask is not None:
            selected_edges = graph.edge_index.T[edges_mask]
            distances = graph.distances[edges_mask]

        edges_plot = go.Scatter3d(
            x=list(
                itertools.chain(
                    *(
                        (graph.coordinates[i, 0], graph.coordinates[j, 0], None)
                        for i, j in selected_edges
                    )
                )
            ),
            y=list(
                itertools.chain(
                    *(
                        (graph.coordinates[i, 1], graph.coordinates[j, 1], None)
                        for i, j in selected_edges
                    )
                )
            ),
            z=list(
                itertools.chain(
                    *(
                        (graph.coordinates[i, 2], graph.coordinates[j, 2], None)
                        for i, j in selected_edges
                    )
                )
            ),
            mode="lines",
            line=dict(
                color=color,
                width=width,
                dash=dash,
            ),
            text=(
                list(
                    itertools.chain(
                        *((f"{d:.3f}Å", f"{d:.3f}Å", None) for d in distances.tolist())
                    )
                )
                if add_annotation
                else None
            ),
            hoverinfo="text",
        )
        return edges_plot
