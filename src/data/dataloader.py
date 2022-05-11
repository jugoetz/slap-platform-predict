import os
from typing import List, Optional, Tuple

import dgl
from dgl.data import DGLDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from src.data.grapher import build_cgr, build_mol_graph
from src.util.configuration import CONFIG


def collate_fn(batch: List[Tuple[dgl.DGLGraph, torch.tensor]]) -> Tuple[dgl.DGLGraph, torch.tensor]:
    """Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch"""

    graphs, labels = map(list, zip(*batch))

    batched_graphs = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)

    return batched_graphs, batched_labels


class SLAPDataModule(pl.LightningDataModule):
    """Data module to load different splits of the SLAP data set."""

    def __init__(self, data_dir, split_dir, cross_validate=None, cv_random_state=42, reaction=True,
                 smiles_columns=("SMILES", ), label_column="targets", batch_size=32):
        """
        Args:
            data_dir: Directory containing data set
            split_dir: Directory containing three files: "train_idx.csv", "val_idx.csv", "test_idx.csv",
                        each containing the indices for the respective split
            cross_validate (int, optional): If int > 1 is passed, use kfold cross-validation where k is this number.
                If None is passed, do not perform cross-validation.
                Note that if performing CV, validation sets will be taken from the train indices. It is not possible to
                pass a separate validation set.
            cv_random_state (int, optional): Random seed for CV. Ineffective if cross_validate is None.
            reaction (bool): Whether data contains reactionSMILES (as opposed to molecule SMILES). Default True
            smiles_columns (Tuple[str]): Headers of columns containing SMILES. Default ("SMILES", )
            label_column (str): Header of column containing labels. Default "targets".
            batch_size (int): Size of {train,val,test} batches. Default 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.cross_validate = cross_validate
        if self.cross_validate in [0, 1]:
            self.cross_validate = None
        self.cv_random_state = cv_random_state
        self.reaction = reaction
        self.smiles_columns = smiles_columns
        self.label_column = label_column
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        full = SLAPDataset(raw_dir=self.data_dir,
                           reaction=self.reaction,
                           smiles_columns=self.smiles_columns,
                           label_column=self.label_column
                           )
        self.atom_feature_size = full.atom_feature_size
        self.bond_feature_size = full.bond_feature_size
        self.feature_size = full.feature_size
        if stage in (None, "fit"):
            idx_train = pd.read_csv(os.path.join(self.split_dir, "train_idx.csv"), header=0).values.flatten()

            if self.cross_validate:
                splitter = KFold(n_splits=self.cross_validate, shuffle=True, random_state=self.cv_random_state)
                idx_cv = [(tr, val) for tr, val in splitter.split(idx_train)]
                self.slap_train = [[full[i] for i in fold[0]] for fold in idx_cv]
                self.slap_val = [[full[i] for i in fold[1]] for fold in idx_cv]
            else:
                idx_val = pd.read_csv(os.path.join(self.split_dir, "val_idx.csv"), header=0).values.flatten()
                self.slap_train = [[full[i] for i in idx_train]]
                self.slap_val = [[full[i] for i in idx_val]]
        if stage in (None, "test"):
            idx_test = pd.read_csv(os.path.join(self.split_dir, "test_idx.csv"), header=0).values.flatten()
            self.slap_test = [full[i] for i in idx_test]

    def train_dataloader(self, fold=0):
        return DataLoader(self.slap_train[fold], batch_size=self.batch_size, collate_fn=collate_fn)

    def val_dataloader(self, fold=0):
        return DataLoader(self.slap_val[fold], batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.slap_test, batch_size=self.batch_size, collate_fn=collate_fn)


class SLAPDataset(DGLDataset):
    """
    SLAP Dataset

    Can load a set of data points containing either a reactionSMILES or a SMILES of a single molecule and one
    column containing labels.

    After processing, the data set will contain a featurized graph encoding of the molecule or reaction.
    If reaction=True, the encoding will be a condensed graph of reaction (CGR).
    """
    def __init__(self,
                 name,
                 raw_dir=None,
                 url=None,
                 reaction=False,
                 smiles_columns=("SMILES", ),
                 label_column="label",
                 save_dir=None,
                 force_reload=False,
                 verbose=True):

        self.reaction = reaction
        self.label_column = label_column
        self.smiles_columns = smiles_columns

        # we hardcode featurizers used for the data set. You could make these part of hyperparameter search of course
        # TODO: set custom Featurizers
        self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="e")
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
            self.graphs = [build_cgr(s, self.atom_featurizer, self.bond_featurizer) for s in smiles]
        else:
            self.graphs = [build_mol_graph(s, self.atom_featurizer, self.bond_featurizer) for s in smiles]

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
        if CONFIG["reaction"]:
            return 2 * n_atom_features  # CGR has 2 x features
        else:
            return n_atom_features

    @property
    def bond_feature_size(self):
        n_bond_features = self.bond_featurizer.feat_size()
        if CONFIG["reaction"]:
            return 2 * n_bond_features  # CGR has 2 x features
        else:
            return n_bond_features

    @property
    def feature_size(self):
        return self.atom_feature_size + self.bond_feature_size

