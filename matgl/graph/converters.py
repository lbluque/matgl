"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""
from __future__ import annotations

import abc

import dgl
import numpy as np
import torch
from dgl.backend import tensor


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[dgl.DGLGraph, list]:
        """Args:
        structure: Input crystals or molecule.

        Returns:
        DGLGraph object, state_attr
        """

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        element_types,
        cart_coords,
        device: str | torch.device = "cpu",
    ) -> tuple[dgl.DGLGraph, list]:
        """Construct a dgl graph from processed structure and bond information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            cart_coords: Cartisian coordinates of all atoms in the structure.

        Returns:
            DGLGraph object, state_attr

        """
        device = torch.device(device)
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl.graph((u, v), num_nodes=len(structure), device=device)
        g.edata["pbc_offset"] = torch.tensor(images, device=device)
        g.edata["lattice"] = torch.tensor(np.repeat(lattice_matrix, g.num_edges(), axis=0), device=device)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], g.edata["lattice"][0])
        g.ndata["node_type"] = torch.tensor(np.hstack([[element_types.index(site.specie.symbol)] for site in structure]), device=device)
        g.ndata["pos"] = torch.tensor(cart_coords, device=device)
        state_attr = [0.0, 0.0]
        return g, state_attr
