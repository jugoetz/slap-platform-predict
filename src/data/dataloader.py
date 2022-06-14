from typing import List, Tuple, Optional

import dgl
from dgl.data import DGLDataset
import torch
import pandas as pd
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from src.data.grapher import build_cgr, build_mol_graph
from src.data.featurizers import ChempropAtomFeaturizer, ChempropBondFeaturizer, RDKit2DGlobalFeaturizer


def collate_fn(batch: List[Tuple[dgl.DGLGraph, Optional[list], torch.tensor]]) -> Tuple[dgl.DGLGraph, Optional[torch.tensor], torch.tensor]:
    """Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch"""

    graphs, global_features, labels = map(list, zip(*batch))

    batched_graphs = dgl.batch(graphs)
    if global_features[0] is None:
        batched_global_features = None
    else:
        batched_global_features = torch.tensor(global_features, dtype=torch.float32)
    batched_labels = torch.tensor(labels)

    return batched_graphs, batched_global_features, batched_labels


class SLAPDataset(DGLDataset):
    """
    SLAP Dataset

    Can load a set of data points containing either a reactionSMILES or a SMILES of a single molecule and one
    column containing labels.

    After processing, the data set will contain a featurized graph encoding of the molecule or reaction.
    If reaction=True, the encoding will be a condensed graph of reaction (CGR).

    If rdkit_features is True, for each molecule, properties are calculated using rdkit. Note that this is mutually
    exclusive with reaction=True.
    """
    def __init__(self,
                 name,
                 raw_dir=None,
                 url=None,
                 reaction=False,
                 rdkit_features=False,
                 graph_type="bond_edges",
                 featurizers="dgllife",
                 smiles_columns=("SMILES", ),
                 label_column="label",
                 save_dir=None,
                 force_reload=False,
                 verbose=True):
        """
        Args:
            name (str): File name of the dataset
            raw_dir (str or path-like): Directory containing data. Default None
            url (str): Url to fetch data from. Default None
            reaction (bool): Whether data is a reaction. If True, data will be loaded as CGR. Default False.
            rdkit_features (bool): Whether to add rdkit features. Default False.
            graph_type (str): Type of graph to use. If "bond_edges", graphs are formed as molecular graphs (nodes are
                    atoms and edges are bonds). These graphs are homogeneous. If "bond_nodes", bond-node graphs will be
                    formed (both atoms and bonds are nodes, edges represent their connectivity).
                    Options: {"bond_edges", "bond_nodes"}. Default "bond_edges".
            featurizers (str): Featurizers to use for atom and bond featurization. Options: {"dgllife", "chemprop"}.
                Default "dgllife".
            smiles_columns (tuple): Headers of columns in data file that contain SMILES strings
            label_column (str): Header of the column in data file that contains the labels
            save_dir (str or path-like): Directory to save the processed data set. If None, `raw_dir` is used. Default None.
            force_reload (bool): Reload data set, ignoring cache. Default False.
            verbose (bool): Whether to provide verbose output
        """

        self.reaction = reaction
        self.label_column = label_column
        self.smiles_columns = smiles_columns
        self.graph_type = graph_type  # whether to form BE- or BN-graph
        self.global_features = None  # container for global features e.g. rdkit

        # featurizer to obtain atom and bond features
        if featurizers == "dgllife":
            self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="e")
        elif featurizers == "chemprop":
            self.atom_featurizer = ChempropAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = ChempropBondFeaturizer(bond_data_field="e")
        else:
            raise ValueError("Unexpected value for 'featurizers'")

        # global featurizer
        if rdkit_features:
            self.global_featurizer = RDKit2DGlobalFeaturizer(normalize=True)

        super(SLAPDataset, self).__init__(name=name,
                                          url=url,
                                          raw_dir=raw_dir,
                                          save_dir=save_dir,
                                          force_reload=force_reload,
                                          verbose=verbose
                                          )

    def process(self):
        """Read data from csv file and generate graphs"""

        csv_data = pd.read_csv(self.raw_path)
        smiles = [csv_data[s] for s in self.smiles_columns]

        # Currently, we don't support having multiple inputs per data point
        if len(smiles) > 1:
            raise NotImplementedError("Multi-input prediction is not implemented.")
        # ...which allows us to do this:
        smiles = smiles[0]

        if self.reaction:
            self.graphs = [build_cgr(s, self.atom_featurizer, self.bond_featurizer, mode="reac_diff", graph_type=self.graph_type) for s in smiles]
        else:
            self.graphs = [build_mol_graph(s, self.atom_featurizer, self.bond_featurizer, graph_type=self.graph_type) for s in smiles]

        if self.global_featurizer:
            if self.reaction:
                # if it is a reaction, we featurize for both reactants, then concatenate
                self.global_features = [[*self.global_featurizer.process(s.split(">>")[0].split(".")[0]), *self.global_featurizer.process(s.split(">>")[0].split(".")[1])] for s in smiles]  # [*l1, *l2] joins lists l1 and l2
            else:
                # if instead we get a single molecule, we just featurize for that
                self.global_features = [self.global_featurizer.process(s) for s in smiles]

        self.labels = csv_data[self.label_column].values.tolist()

    def __getitem__(self, idx):
        """ Get graph and label by index

        Args:
            idx (int): Item index

        Returns:
            (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.global_features[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    @property
    def atom_feature_size(self):
        n_atom_features = self.atom_featurizer.feat_size()
        if self.reaction:
            return 2 * n_atom_features  # CGR has 2 x features
        else:
            return n_atom_features

    @property
    def bond_feature_size(self):
        n_bond_features = self.bond_featurizer.feat_size()
        if self.reaction:
            return 2 * n_bond_features  # CGR has 2 x features
        else:
            return n_bond_features

    @property
    def feature_size(self):
        return self.atom_feature_size + self.bond_feature_size

    @property
    def global_feature_size(self):
        if self.global_featurizer:
            n_global_features = self.global_featurizer.feat_size
            if self.reaction:
                return 2 * n_global_features  # for 2 reactants we have 2 x features
            else:
                return n_global_features
        else:
            return 0
