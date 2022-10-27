import os
from typing import List, Tuple, Optional, Union

import dgl
import numpy as np
from dgl.data import DGLDataset
import torch
import pandas as pd
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from src.data.grapher import build_cgr, build_mol_graph
from src.data.featurizers import (
    ChempropAtomFeaturizer,
    ChempropBondFeaturizer,
    SLAPAtomFeaturizer,
    SLAPBondFeaturizer,
    RDKit2DGlobalFeaturizer,
    RDKitMorganFingerprinter,
    OneHotEncoder,
    FromFileFeaturizer,
)


def collate_fn(
    batch: List[Tuple[dgl.DGLGraph, list, torch.tensor]]
) -> Tuple[dgl.DGLGraph, Optional[torch.tensor], torch.tensor]:
    """Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch"""

    graphs, global_features, labels = map(list, zip(*batch))

    batched_graphs = dgl.batch(graphs)
    if len(global_features[0]) == 0:
        batched_global_features = None
    else:
        batched_global_features = torch.tensor(global_features, dtype=torch.float32)

    if labels[0] is None:
        batched_labels = None
    else:
        batched_labels = torch.tensor(labels)

    return batched_graphs, batched_global_features, batched_labels


class SLAPDataset(DGLDataset):
    """
    SLAP Dataset

    Can load a set of data points containing either a reactionSMILES or a SMILES of a single molecule and one
    column containing labels.

    After processing, the data set will contain a featurized graph encoding of the molecule or reaction.
    If reaction=True, the encoding will be a condensed graph of reaction (CGR).
    """

    def __init__(
        self,
        name: str,
        raw_dir: Union[str, os.PathLike] = None,
        url: str = None,
        reaction=False,
        global_features: Union[str, List[str]] = None,
        global_features_file: Union[str, os.PathLike] = None,
        graph_type: str = "bond_edges",
        featurizers: str = "dgllife",
        smiles_columns: tuple = ("SMILES",),
        label_column: Optional[str] = "label",
        save_dir: Union[str, os.PathLike] = None,
        force_reload=False,
        verbose=True,
    ):
        """
        Args:
            name: File name of the dataset
            raw_dir: Directory containing data. Default None
            url: Url to fetch data from. Default None
            reaction: Whether data is a reaction. If True, data will be loaded as CGR. Default False.
            global_features: Which global features to add.
                Options: {"RDKit", "FP", "OHE", "fromFile", None} and combinations thereof.
                If a string is passed, any way to combine the features is allowed (e.g. "RDKit_FP", "RDKit+FP",...)
                If a list is passed, only the strings list in options as list members will have an effect.
                Passing and empty list, or a list with unrecognized options is equivalent to passing None.
                Default None.
            global_features_file: Path to file containing global features.
                Only used with global_features=fromFile. Default None.
            graph_type: Type of graph to use. If "bond_edges", graphs are formed as molecular graphs (nodes are
                    atoms and edges are bonds). These graphs are homogeneous. If "bond_nodes", bond-node graphs will be
                    formed (both atoms and bonds are nodes, edges represent their connectivity).
                    Options: {"bond_edges", "bond_nodes"}. Default "bond_edges".
            featurizers: Featurizers to use for atom and bond featurization. Options: {"dgllife", "chemprop"}.
                Default "dgllife".
            smiles_columns: Headers of columns in data file that contain SMILES strings
            label_column: Header of the column in data file that contains the labels
            save_dir: Directory to save the processed data set. If None, `raw_dir` is used. Default None.
            force_reload: Reload data set, ignoring cache. Default False.
            verbose: Whether to provide verbose output
        """

        if global_features is None:
            global_features = []

        self.reaction = reaction
        self.label_column = label_column
        self.smiles_columns = smiles_columns
        self.graph_type = graph_type  # whether to form BE- or BN-graph
        self.global_features = (
            []
        )  # container for global features e.g. rdkit or fingerprints
        self.global_featurizers = []  # container for global featurizers

        # featurizer to obtain atom and bond features
        if featurizers == "dgllife":
            self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="e")
        elif featurizers == "chemprop":
            self.atom_featurizer = ChempropAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = ChempropBondFeaturizer(bond_data_field="e")
        elif featurizers == "custom":
            self.atom_featurizer = SLAPAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = SLAPBondFeaturizer(bond_data_field="e")
        else:
            raise ValueError("Unexpected value for 'featurizers'")

        # global featurizer(s)
        if "RDKit" in global_features:
            self.global_featurizers.append(RDKit2DGlobalFeaturizer(normalize=True))
        if "FP" in global_features:
            self.global_featurizers.append(
                RDKitMorganFingerprinter(radius=6, n_bits=1024)
            )
        if "OHE" in global_features:
            self.global_featurizers.append(OneHotEncoder())
        if "fromFile" in global_features:
            self.global_featurizers.append(
                FromFileFeaturizer(filename=global_features_file)
            )

        super(SLAPDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
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

        # clear previous data
        self.graphs = []
        self.global_features = [np.array(()) for _ in smiles]
        self.labels = []

        if self.reaction:
            self.graphs = [
                build_cgr(
                    s,
                    self.atom_featurizer,
                    self.bond_featurizer,
                    mode="reac_diff",
                    graph_type=self.graph_type,
                )
                for s in smiles
            ]

        else:
            self.graphs = [
                build_mol_graph(
                    s,
                    self.atom_featurizer,
                    self.bond_featurizer,
                    graph_type=self.graph_type,
                )
                for s in smiles
            ]

        if len(self.global_featurizers) > 0:
            if self.reaction:
                # if it is a reaction, we featurize for both reactants, then concatenate
                for global_featurizer in self.global_featurizers:
                    if isinstance(global_featurizer, OneHotEncoder):
                        # for OHE, we need to set up the encoder with the list(s) of smiles it should encode
                        smiles_reactant1 = [s.split(".")[0] for s in smiles]
                        smiles_reactant2 = [
                            s.split(">>")[0].split(".")[1] for s in smiles
                        ]
                        global_featurizer.add_dimension(smiles_reactant1)
                        global_featurizer.add_dimension(smiles_reactant2)
                    self.global_features = [
                        np.concatenate(i)
                        for i in zip(
                            self.global_features,
                            [
                                global_featurizer.process(*s.split(">>")[0].split("."))
                                for s in smiles
                            ],
                        )
                    ]

            else:
                # if instead we get a single molecule, we just featurize for that
                for global_featurizer in self.global_featurizers:
                    if isinstance(global_featurizer, OneHotEncoder):
                        # for OHE, we need to set up the encoder with the list(s) of smiles it should encode
                        global_featurizer.add_dimension(smiles)

                    self.global_features = [
                        np.concatenate(i)
                        for i in zip(
                            self.global_features,
                            [global_featurizer.process(s) for s in smiles],
                        )
                    ]

        if self.label_column:
            self.labels = csv_data[self.label_column].values.tolist()
        else:
            # allow having no labels, e.g. for prediction
            self.labels = [None for _ in smiles]

        # little safety net
        assert len(self.graphs) == len(self.labels) == len(self.global_features)

    def __getitem__(self, idx):
        """Get graph and label by index

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
        n_global_features = 0
        for global_featurizer in self.global_featurizers:
            if self.reaction and not isinstance(global_featurizer, OneHotEncoder):
                # for 2 reactants we have 2 x features (except for the OHE which always encorporates all inputs in feat size)
                n_global_features += 2 * global_featurizer.feat_size
            else:
                n_global_features += global_featurizer.feat_size
        return n_global_features
