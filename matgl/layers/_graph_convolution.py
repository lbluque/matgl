"""Graph convolution layer (GCL) implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dgl
import dgl.function as fn
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Identity, Module

from matgl.layers._core import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence


class MEGNetGraphConv(Module):
    """A MEGNet graph convolution layer in DGL."""

    def __init__(
        self,
        edge_func: Module,
        node_func: Module,
        state_func: Module,
    ) -> None:
        """
        Args:
            edge_func: Edge update function.
            node_func: Node update function.
            state_func: Global state update function.
        """
        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.state_func = state_func

    @staticmethod
    def from_dims(
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int],
        activation: Module,
    ) -> MEGNetGraphConv:
        """Create a MEGNet graph convolution layer from dimensions.

        Args:
            edge_dims (list[int]): Edge dimensions.
            node_dims (list[int]): Node dimensions.
            state_dims (list[int]): State dimensions.
            activation (Module): Activation function.

        Returns:
            MEGNetGraphConv: MEGNet graph convolution layer.
        """
        # TODO(marcel): Softplus doesn't exactly match paper's SoftPlus2
        # TODO(marcel): Should we activate last?
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        attr_update = MLP(state_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, attr_update)

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = edges.src["u"]
        eij = edges.data.pop("e")
        inputs = torch.hstack([vi, vj, eij, u])
        mij = {"mij": self.edge_func(inputs)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update.

        Args:
            graph: Input graph

        Returns:
            Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        graph.edata["e"] = graph.edata.pop("mij")
        return graph.edata["e"]

    def node_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform node update.

        Args:
            graph: Input graph

        Returns:
            Output tensor for nodes.
        """
        graph.update_all(fn.copy_e("e", "e"), fn.mean("e", "ve"))
        ve = graph.ndata.pop("ve")
        v = graph.ndata.pop("v")
        u = graph.ndata.pop("u")
        inputs = torch.hstack([v, ve, u])
        graph.ndata["v"] = self.node_func(inputs)
        return graph.ndata["v"]

    def state_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: Input graph
            state_feat: Input attributes

        Returns:
            Output tensor for attributes
        """
        u_edge = dgl.readout_edges(graph, feat="e", op="mean")
        u_vertex = dgl.readout_nodes(graph, feat="v", op="mean")
        u_edge = torch.squeeze(u_edge)
        u_vertex = torch.squeeze(u_vertex)
        inputs = torch.hstack([state_feat.squeeze(), u_edge, u_vertex])
        state_feat = self.state_func(inputs)
        return state_feat

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->attribute updates.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state)

        Returns:
            (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            graph.ndata["u"] = dgl.broadcast_nodes(graph, state_feat)

            edge_feat = self.edge_update_(graph)
            node_feat = self.node_update_(graph)
            state_feat = self.state_update_(graph, state_feat)

        return edge_feat, node_feat, state_feat


class MEGNetBlock(Module):
    """A MEGNet block comprising a sequence of update operations."""

    def __init__(
        self, dims: list[int], conv_hiddens: list[int], act: Module, dropout: float | None = None, skip: bool = True
    ) -> None:
        """
        Init the MEGNet block with key parameters.

        Args:
            dims: Dimension of dense layers before graph convolution.
            conv_hiddens: Architecture of hidden layers of graph convolution.
            act: Activation type.
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according
                to a Bernoulli distribution.
            skip: Residual block.
        """
        super().__init__()
        self.has_dense = len(dims) > 1
        self.activation = act
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]

        mlp_kwargs = {
            "dims": dims,
            "activation": self.activation,
            "activate_last": True,
            "bias_last": True,
        }
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore
        self.state_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore

        # compute input sizes
        edge_in = 2 * conv_dim + conv_dim + conv_dim  # 2*NDIM+EDIM+GDIM
        node_in = out_dim + conv_dim + conv_dim  # EDIM+NDIM+GDIM
        attr_in = out_dim + out_dim + conv_dim  # EDIM+NDIM+GDIM
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in, *conv_hiddens],
            node_dims=[node_in, *conv_hiddens],
            state_dims=[attr_in, *conv_hiddens],
            activation=self.activation,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout
        self.skip = skip

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """MEGNetBlock forward pass.

        Args:
            graph (dgl.DGLGraph): A DGLGraph.
            edge_feat (Tensor): Edge features.
            node_feat (Tensor): Node features.
            state_feat (Tensor): Graph attributes (global state).

        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated (edge features,
                node features, graph attributes)
        """
        inputs = (edge_feat, node_feat, state_feat)
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_feat = self.state_func(state_feat)

        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat, node_feat, state_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            state_feat = self.dropout(state_feat)  # pylint: disable=E1102

        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_feat = state_feat + inputs[2]

        return edge_feat, node_feat, state_feat


class M3GNetGraphConv(Module):
    """A M3GNet graph convolution layer in DGL."""

    def __init__(
        self,
        include_states: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ):
        """Parameters:
        include_state (bool): Whether including state
        edge_update_func (Module): Update function for edges (Eq. 4)
        edge_weight_func (Module): Weight function for radial basis functions (Eq. 4)
        node_update_func (Module): Update function for nodes (Eq. 5)
        node_weight_func (Module): Weight function for radial basis functions (Eq. 5)
        attr_update_func (Module): Update function for state feats (Eq. 6).
        """
        super().__init__()
        self.include_states = include_states
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree,
        include_states,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: Module,
    ) -> M3GNetGraphConv:
        """M3GNetGraphConv initialization.

        Args:
            degree (int): max_n*max_l
            include_states (bool): whether including state or not
            edge_dims (list): NN architecture for edge update function
            node_dims (list): NN architecture for node update function
            state_dims (list): NN architecture for state update function
            activation (nn.Nodule): activation function

        Returns:
        M3GNetGraphConv (class)
        """
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        attr_update_func = MLP(state_dims, activation, activate_last=True) if include_states else None  # type: ignore
        return M3GNetGraphConv(
            include_states, edge_update_func, edge_weight_func, node_update_func, node_weight_func, attr_update_func
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        """Edge update functions.

        Args:
        edges (DGL graph): edges in dgl graph

        Returns:
        mij: message passing between node i and j
        """
        vi = edges.src["v"]
        vj = edges.dst["v"]
        if self.include_states:
            u = edges.src["u"]
        eij = edges.data.pop("e")
        rbf = edges.data["rbf"]
        rbf = rbf.float()
        inputs = torch.hstack([vi, vj, eij, u]) if self.include_states else torch.hstack([vi, vj, eij])
        mij = {"mij": self.edge_update_func(inputs) * self.edge_weight_func(rbf)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update.

        Args:
        graph: DGL graph

        Returns:
        edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata.pop("mij")
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform node update.

        Args:
            graph: DGL graph
            state_feat: State attributes

        Returns:
            node_update: node features update
        """
        eij = graph.edata["e"]
        src_id = graph.edges()[0]
        vi = graph.ndata["v"][src_id]
        dst_id = graph.edges()[1]
        vj = graph.ndata["v"][dst_id]
        rbf = graph.edata["rbf"]
        rbf = rbf.float()
        if self.include_states:
            u = dgl.broadcast_edges(graph, state_feat)
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        graph.edata["mess"] = self.node_update_func(inputs) * self.node_weight_func(rbf)
        graph.update_all(fn.copy_e("mess", "mess"), fn.sum("mess", "ve"))
        node_update = graph.ndata.pop("ve")
        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: DGL graph
            state_feat: graph features

        Returns:
        state_update: state_features update
        """
        u = state_feat
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, uv])
        state_feat = self.state_update_func(inputs)  # type: ignore
        return state_feat

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Perform sequence of edge->node->states updates.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state).

        Returns:
            (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            if self.include_states:
                graph.ndata["u"] = dgl.broadcast_nodes(graph, state_feat)

            edge_update = self.edge_update_(graph)
            graph.edata["e"] = edge_feat + edge_update
            node_update = self.node_update_(graph, state_feat)
            graph.ndata["v"] = node_feat + node_update
            if self.include_states:
                state_feat = self.state_update_(graph, state_feat)

        return edge_feat + edge_update, node_feat + node_update, state_feat


class M3GNetBlock(Module):
    """A M3GNet block comprising a sequence of update operations."""

    def __init__(
        self,
        degree: int,
        activation: Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        dropout: float | None = None,
    ) -> None:
        """

        Args:
            degree: Number of radial basis functions
            activation: activation
            dim_node_feats: Number of node features
            dim_edge_feats: Number of edge features
            dim_state_feats: Number of state features
            conv_hiddens: Dimension of hidden layers
            activation: Activation type
            include_state: Including state features or not
            dropout: Probability of an element to be zero in dropout layer.
        """
        super().__init__()

        self.activation = activation

        # compute input sizes
        if include_state:
            edge_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats  # type: ignore
            node_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats  # type: ignore
            attr_in = dim_node_feats + dim_state_feats  # type: ignore
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=[attr_in, *conv_hiddens, dim_state_feats],  # type: ignore
                activation=self.activation,
            )
        else:
            edge_in = 2 * dim_node_feats + dim_edge_feats  # 2*NDIM+EDIM
            node_in = 2 * dim_node_feats + dim_edge_feats  # 2*NDIM+EDIM
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=None,  # type: ignore
                activation=self.activation,
            )

        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple:
        """
        Args:
            graph: DGL graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features.

        Returns:
            A tuple of updated features
        """
        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat, node_feat, state_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            if state_feat is not None:
                state_feat = self.dropout(state_feat)  # pylint: disable=E1102

        return edge_feat, node_feat, state_feat


class CHGNetGraphConv(nn.Module):
    """A CHGNet atom graph convolution layer in DGL."""

    def __init__(
        self,
        node_update_func: Module,
        node_out_func: Module,
        edge_update_func: Module | None,
        node_weight_func: Module | None,
        edge_weight_func: Module | None,
        state_update_func: Module | None,
    ):
        """
        Args:
            include_state: Whether including state
            node_update_func: Update function for message between nodes (atoms)
            node_out_func: Output function for nodes (atoms), after message aggregation
            edge_update_func: Update function for edges (bonds). If None is given, the
                edges are not updated.
            node_weight_func: Weight function for radial basis functions.
                If None is given, no layer-wise weights will be used.
            edge_weight_func: Weight function for radial basis functions
                If None is given, no layer-wise weights will be used.
            state_update_func: Update function for state feats.
        """
        super().__init__()
        self.include_state = state_update_func is not None
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_out_func = node_out_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @classmethod
    def from_dims(
        cls,
        activation: Module,
        node_dims: Sequence[int],
        edge_dims: Sequence[int] | None = None,
        state_dims: Sequence[int] | None = None,
        rbf_order: int = 0,
    ) -> CHGNetGraphConv:
        """Create a CHGNetAtomGraphConv layer from dimensions.

        Args:
            activation: activation function
            node_dims: NN architecture for node update function given as a list of
                dimensions of each layer.
            edge_dims: NN architecture for edge update function given as a list of
                dimensions of each layer.
            state_dims: NN architecture for state update function given as a list of
                dimensions of each layer.
            rbf_order: RBF order specifying input dimensions for linear layer
                specifying message weights. If 0, no layer-wise weights are used.

        Returns:
            CHGNetAtomGraphConv
        """
        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_out_func = nn.Linear(in_features=node_dims[-1], out_features=node_dims[-1], bias=False)
        node_weight_func = (
            nn.Linear(in_features=rbf_order, out_features=node_dims[-1], bias=False) if rbf_order > 0 else None
        )
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:]) if edge_dims is not None else None
        edge_weight_func = (
            nn.Linear(in_features=rbf_order, out_features=edge_dims[-1], bias=False)
            if rbf_order > 0 and edge_dims is not None
            else None
        )
        state_update_func = MLP(state_dims, activation, activate_last=True) if state_dims is not None else None

        return cls(
            node_update_func=node_update_func,
            node_out_func=node_out_func,
            edge_update_func=edge_update_func,
            node_weight_func=node_weight_func,
            edge_weight_func=edge_weight_func,
            state_update_func=state_update_func,
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, Tensor]:
        """Edge user defined update function.

        Update for bond features (edges) in atom graph.

        Args:
            edges: edges in atom graph (ie bonds)

        Returns:
            edge_update: edge features update
        """
        atom_i = edges.src["features"]  # first atom features
        atom_j = edges.dst["features"]  # second atom features
        bond_ij = edges.data["features"]  # bond features

        if self.include_state:
            global_state = edges.data["global_state"]
            inputs = torch.hstack([atom_i, bond_ij, atom_j, global_state])
        else:
            inputs = torch.hstack([atom_i, bond_ij, atom_j])

        edge_update = self.edge_update_func(inputs)  # type: ignore
        if self.edge_weight_func is not None:
            rbf = edges.data["bond_expansion"]
            rbf = rbf.float()
            edge_update = edge_update * self.edge_weight_func(rbf)

        return {"feat_update": edge_update}

    def edge_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor | None) -> Tensor:
        """Perform edge update -> bond features.

        Args:
            graph: atom graph
            shared_weights: atom graph edge weights shared between convolution layers

        Returns:
            edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata["feat_update"]
        if shared_weights is not None:
            edge_update = edge_update * shared_weights
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor | None) -> Tensor:
        """Perform node update -> atom features.

        Args:
            graph: DGL atom graph
            shared_weights: node message shared weights

        Returns:
            node_update: updated node features
        """
        src, dst = graph.edges()
        atom_i = graph.ndata["features"][src]  # first atom features
        atom_j = graph.ndata["features"][dst]  # second atom features
        bond_ij = graph.edata["features"]  # bond features

        if self.include_state:
            global_state = graph.edata["global_state"]
            inputs = torch.hstack([atom_i, bond_ij, atom_j, global_state])
        else:
            inputs = torch.hstack([atom_i, bond_ij, atom_j])

        messages = self.node_update_func(inputs)

        # smooth out the messages with layer-wise weights
        if self.node_weight_func is not None:
            rbf = graph.edata["bond_expansion"]
            rbf = rbf.float()
            messages = messages * self.node_weight_func(rbf)
        # smooth out the messages with shared weights
        if shared_weights is not None:
            messages = messages * shared_weights

        # message passing
        graph.edata["message"] = messages
        graph.update_all(fn.copy_e("message", "message"), fn.sum("message", "feat_update"))

        # update nodes
        node_update = self.node_out_func(graph.ndata["feat_update"])  # the bond update

        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_attr: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: atom graph
            state_attr: global state features

        Returns:
            state_update: state features update
        """
        node_avg = dgl.readout_nodes(graph, feat="features", op="mean")
        inputs = torch.hstack([state_attr, node_avg])
        state_attr = self.state_update_func(inputs)  # type: ignore
        return state_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        state_attr: Tensor,
        shared_node_weights: Tensor | None,
        shared_edge_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        Args:
            graph: atom graph
            node_features: node features
            edge_features: edge features
            state_attr: state attributes
            shared_node_weights: shared node message weights
            shared_edge_weights: shared edge message weights

        Returns:
            tuple: updated node features, updated edge features, updated state attributes
        """
        with graph.local_scope():
            graph.ndata["features"] = node_features
            graph.edata["features"] = edge_features

            if self.include_state:
                graph.edata["global_state"] = dgl.broadcast_edges(graph, state_attr)

            if self.edge_update_func is not None:
                edge_update = self.edge_update_(graph, shared_edge_weights)
                new_edge_features = edge_features + edge_update
                graph.edata["features"] = new_edge_features
            else:
                new_edge_features = edge_features

            node_update = self.node_update_(graph, shared_node_weights)
            new_node_features = node_features + node_update
            graph.ndata["features"] = new_node_features

            if self.include_state:
                state_attr = self.state_update_(graph, state_attr)  # type: ignore

        return new_node_features, new_edge_features, state_attr


class CHGNetAtomGraphBlock(nn.Module):
    """A CHGNet atom graph block as a sequence of operations involving a message passing layer over the atom graph."""

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        activation: Module,
        atom_hidden_dims: Sequence[int],
        bond_hidden_dims: Sequence[int] | None = None,
        num_state_feats: int | None = None,
        rbf_order: int = 0,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_atom_feats: number of atom features
            num_bond_feats: number of bond features
            activation: activation function
            atom_hidden_dims: dimensions of atom convolution hidden layers
            bond_hidden_dims: dimensions of bond update hidden layers.
            num_state_feats: number of state features if include_state is True
            rbf_order: whether to include layer-wise node weights
             RBF order specifying input dimensions for linear layer specifying message weights.
                If 0, no layer-wise weights are used
            dropout: dropout probability.
        """
        super().__init__()

        node_input_dim = 2 * num_atom_feats + num_bond_feats
        if num_state_feats is not None:
            node_input_dim += num_state_feats
            state_dims = [num_atom_feats + num_state_feats, *atom_hidden_dims, num_state_feats]
        else:
            state_dims = None
        node_dims = [node_input_dim, *atom_hidden_dims, num_atom_feats]
        edge_dims = [node_input_dim, *bond_hidden_dims, num_bond_feats] if bond_hidden_dims is not None else None

        self.conv_layer = CHGNetGraphConv.from_dims(
            activation=activation,
            node_dims=node_dims,
            edge_dims=edge_dims,
            state_dims=state_dims,
            rbf_order=rbf_order,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_features: Tensor,
        bond_features: Tensor,
        state_attr: Tensor,
        shared_node_weights: Tensor | None,
        shared_edge_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of bond(optional)->atom->states(optional) updates.

        Args:
            graph: atom graph
            atom_features: node features
            bond_features: edge features
            state_attr: state attributes
            shared_node_weights: node message weights shared amongst layers
            shared_edge_weights: edge message weights shared amongst layers
        """
        atom_features, bond_features, state_attr = self.conv_layer(
            graph=graph,
            node_features=atom_features,
            edge_features=bond_features,
            state_attr=state_attr,
            shared_node_weights=shared_node_weights,
            shared_edge_weights=shared_edge_weights,
        )
        # move skip connections here? dropout before skip connections?
        atom_features = self.dropout(atom_features)
        bond_features = self.dropout(bond_features)
        if state_attr is not None:
            state_attr = self.dropout(state_attr)

        return atom_features, bond_features, state_attr


class CHGNetLineGraphConv(nn.Module):
    """A CHGNet atom graph convolution layer in DGL.

    This implements both the bond and angle update functions in the CHGNet paper
    as line graph updates.
    """

    def __init__(
        self,
        node_update_func: Module,
        node_out_func: Module,
        edge_update_func: Module | None,
        node_weight_func: Module | None,
    ):
        """
        Args:
            node_update_func: Update function for message between nodes (bonds)
            node_out_func: Output function for nodes (bonds), after message aggregation
            edge_update_func: edge update function (for angle features)
            node_weight_func: layer node weight function.
        """
        super().__init__()

        self.node_update_func = node_update_func
        self.node_out_func = node_out_func
        self.node_weight_func = node_weight_func
        self.edge_update_func = edge_update_func

    @classmethod
    def from_dims(
        cls,
        node_dims: list[int],
        edge_dims: list[int] | None = None,
        node_weight_input_dims: int = 0,
    ) -> CHGNetLineGraphConv:
        """
        Args:
            node_dims: NN architecture for node update function given as a list of
                dimensions of each layer.
            edge_dims: NN architecture for edge update function given as a list of
                dimensions of each layer.
            node_weight_input_dims: input dimensions for linear layer of node weights.
                (the RBF order)
                If 0, no layer-wise weights are used.

        Returns:
            CHGNetBondGraphConv
        """
        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_out_func = nn.Linear(in_features=node_dims[-1], out_features=node_dims[-1], bias=False)
        node_weight_func = nn.Linear(node_weight_input_dims, node_dims[-1]) if node_weight_input_dims > 0 else None
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:]) if edge_dims is not None else None

        return cls(
            node_update_func=node_update_func,
            node_out_func=node_out_func,
            edge_update_func=edge_update_func,
            node_weight_func=node_weight_func,
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, Tensor]:
        """Edge user defined update function.

        Update angle features (edges in bond graph)

        Args:
            edges: edge batch

        Returns:
            edge_update: edge features update
        """
        bonds_i = edges.src["features"]  # first bonds features
        bonds_j = edges.dst["features"]  # second bonds features
        angle_ij = edges.data["features"]
        atom_ij = edges.data["aux_features"]  # center atom features
        inputs = torch.hstack([bonds_i, angle_ij, atom_ij, bonds_j])
        messages_ij = self.edge_update_func(inputs)  # type: ignore
        return {"feat_update": messages_ij}

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update -> update angle features.

        Args:
            graph: bond graph (line graph of atom graph)

        Returns:
            edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata["feat_update"]
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor | None) -> Tensor:
        """Perform node update -> update bond features.

        Args:
            graph: bond graph (line graph of atom graph)
            shared_weights: node message shared weights

        Returns:
            node_update: bond features update
        """
        src, dst = graph.edges()
        bonds_i = graph.ndata["features"][src]  # first bond feature
        bonds_j = graph.ndata["features"][dst]  # second bond feature
        angle_ij = graph.edata["features"]
        atom_ij = graph.edata["aux_features"]  # center atom features
        inputs = torch.hstack([bonds_i, angle_ij, atom_ij, bonds_j])

        messages = self.node_update_func(inputs)

        # smooth out messages with layer-wise weights
        if self.node_weight_func is not None:
            rbf = graph.ndata["bond_expansion"]
            weights = self.node_weight_func(rbf)
            weights_i, weights_j = weights[src], weights[dst]
            messages = messages * weights_i * weights_j

        # smooth out messages with shared weights
        if shared_weights is not None:
            weights_i, weights_j = shared_weights[src], shared_weights[dst]
            messages = messages * weights_i * weights_j

        # message passing
        graph.edata["message"] = messages
        graph.update_all(fn.copy_e("message", "message"), fn.sum("message", "feat_update"))

        # update nodes
        node_update = self.node_out_func(graph.ndata["feat_update"])  # the bond update
        return node_update

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        aux_edge_features: Tensor,
        shared_node_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        Args:
            graph: bond graph (line graph of atom graph)
            node_features: bond features (edge features (for bonds within three body cutoff in atom graph)
            edge_features: angle features (edge features to be updated)
            aux_edge_features: center atom features (edge features that are not updated)

            shared_node_weights: shared node message weights

        Returns:
            tuple: update edge features, update node features
            note that the node features are the bond features included in the line graph only.
        """
        with graph.local_scope():
            graph.ndata["features"] = node_features
            graph.edata["features"] = edge_features
            graph.edata["aux_features"] = aux_edge_features

            # node (bond) update
            node_update = self.node_update_(graph, shared_node_weights)
            new_node_features = node_features + node_update
            graph.ndata["features"] = new_node_features

            # edge (angle) update (should angle update be done before node update?)
            if self.edge_update_func is not None:
                edge_update = self.edge_update_(graph)
                new_edge_features = edge_features + edge_update
                graph.edata["features"] = new_edge_features
            else:
                new_edge_features = edge_features

        return new_node_features, new_edge_features


class CHGNetBondGraphBlock(nn.Module):
    """A CHGNet atom graph block as a sequence of operations involving a message passing layer over the bond graph."""

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        num_angle_feats: int,
        bond_hidden_dims: Sequence[int],
        angle_hidden_dims: Sequence[int] | None,
        rbf_order: int = 0,
        bond_dropout: float = 0.0,
        angle_dropout: float = 0.0,
    ):
        """
        Args:
            num_atom_feats: number of atom features
            num_bond_feats: number of bond features
            num_angle_feats: number of angle features
            bond_hidden_dims: dimensions of hidden layers of bond graph convolution
            angle_hidden_dims: dimensions of hidden layers of angle update function
            rbf_order: dimensions of input to node weight function (num RBF functions)
                If 0, no layer-wise node weights are used.
            bond_dropout: dropout probability for bond graph convolution.
            angle_dropout: dropout probability for angle update function.
        """
        super().__init__()

        node_input_dim = 2 * num_bond_feats + num_angle_feats + num_atom_feats
        node_dims = [node_input_dim, *bond_hidden_dims, num_bond_feats]
        edge_dims = [node_input_dim, *angle_hidden_dims, num_angle_feats] if angle_hidden_dims is not None else None

        self.conv_layer = CHGNetLineGraphConv.from_dims(
            node_dims=node_dims,
            edge_dims=edge_dims,
            node_weight_input_dims=rbf_order,
        )

        self.bond_dropout = nn.Dropout(bond_dropout) if bond_dropout > 0.0 else nn.Identity()
        self.angle_dropout = nn.Dropout(angle_dropout) if angle_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_features: Tensor,
        bond_features: Tensor,
        angle_features: Tensor,
        shared_node_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Perform convolution in BondGraph to update bond and angle features.

        Args:
            graph: bond graph (line graph of atom graph)
            atom_features: atom features
            bond_features: bond features
            angle_features: concatenated center atom and angle features
            shared_node_weights: shared node message weights

        Returns:
            tuple: update bond features, update angle features
        """
        node_features = bond_features[graph.ndata["bond_index"]]
        edge_features = angle_features
        aux_edge_features = atom_features[graph.edata["center_atom_index"]]

        bond_features_, angle_features = self.conv_layer(
            graph, node_features, edge_features, aux_edge_features, shared_node_weights
        )

        bond_features_ = self.bond_dropout(bond_features_)
        angle_features = self.angle_dropout(angle_features)
        bond_features[graph.ndata["bond_index"]] = bond_features_

        return bond_features, angle_features
