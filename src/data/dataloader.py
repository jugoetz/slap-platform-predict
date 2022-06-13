from typing import List, Tuple

import dgl
from dgl.data import DGLDataset
import torch
import pandas as pd
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from src.data.grapher import build_cgr, build_mol_graph
from src.data.featurizers import ChempropAtomFeaturizer, ChempropBondFeaturizer


def collate_fn(batch: List[Tuple[dgl.DGLGraph, torch.tensor]]) -> Tuple[dgl.DGLGraph, torch.tensor]:
    """Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch"""

    graphs, labels = map(list, zip(*batch))

    batched_graphs = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)

    return batched_graphs, batched_labels


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
                 molecular_graph=True,
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
            molecular_graph (bool): If True, graphs are formed as molecular graphs (nodes are atoms and edges are
                bonds). Else, bond-node graphs will be formed (both atoms and bonds are nodes, edges represent their
                connectivity). Default True.
            featurizers (str): Featurizers to use for atom and bond featurization. Options: {"dgllife", "chemprop"}.
                Default "dgllife".
            smiles_columns (tuple): Headers of columns in data file that contain SMILES strings
            label_column (str): Header of the column in data file that contains the labels
            save_dir (str or path-like): Directory to save the processed data set. If None, `raw_dir` is used. Default None.
            force_reload (bool): Reload data set, ignoring cache. Default False.
            verbose (bool): Whether to provide verbose output
        """

        if rdkit_features and reaction:
            raise ValueError("Cannot use rdkit features with reaction input")

        self.reaction = reaction
        self.rdkit_features = rdkit_features  # TODO implement functionality
        self.label_column = label_column
        self.smiles_columns = smiles_columns
        self.molecular_graph = molecular_graph

        # we hardcode featurizers used for the data set. You could make these part of hyperparameter search of course
        if featurizers == "dgllife":
            self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="e")
        elif featurizers == "chemprop":
            self.atom_featurizer = ChempropAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = ChempropBondFeaturizer(bond_data_field="e")
        else:
            raise ValueError("Unexpected value for 'featurizers'")
        self.global_featurizer = None

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
            self.graphs = [build_cgr(s, self.atom_featurizer, self.bond_featurizer, mode="reac_diff", molecular_graph=self.molecular_graph) for s in smiles]
        else:
            self.graphs = [build_mol_graph(s, self.atom_featurizer, self.bond_featurizer, molecular_graph=self.molecular_graph) for s in smiles]

        self.labels = csv_data[self.label_column].values.tolist()

    def __getitem__(self, idx):
        """ Get graph and label by index

        Args:
            idx (int): Item index

        Returns:
            (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

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

