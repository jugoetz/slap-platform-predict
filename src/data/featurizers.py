from functools import partial

import numpy as np
import torch
from dgllife.utils.featurizers import (
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    ConcatFeaturizer,
    atom_type_one_hot,
    atom_total_degree_one_hot,
    atom_formal_charge_one_hot,
    atom_chiral_tag_one_hot,
    atom_total_num_H_one_hot,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_mass,
    bond_type_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot
)
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit.DataStructs import ConvertToNumpyArray


class ChempropAtomFeaturizer(BaseAtomFeaturizer):
    """A DGLLife implementation of the atom featurizer used in Chemprop.

    The atom features include:

    * **One hot encoding of the atom type by atomic number**. Atomic numbers 1 - 100 are supported.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 6``.
    * **Formal charge of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    * **Mass of the atom**. Divided by 100, not onehot encoded.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    """

    def __init__(self, atom_data_field='h'):
        allowable_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                           'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                           'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                           'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                           'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                           'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
                           'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm']

        super(ChempropAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=allowable_atoms, encode_unknown=True),
                 partial(atom_total_degree_one_hot, allowable_set=list(range(6)), encode_unknown=True),
                 partial(atom_formal_charge_one_hot, allowable_set=[-1, -2, 1, 2, 0], encode_unknown=True),
                 partial(atom_chiral_tag_one_hot, encode_unknown=True),
                 # note that this encode_unknown=True does not make sense as the chiral tags already cover this case. But we follow the ref implementation.
                 partial(atom_total_num_H_one_hot, allowable_set=list(range(5)), encode_unknown=True),
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_mass,
                 ]
            )})


class ChempropBondFeaturizer(BaseBondFeaturizer):
    """A DGLLife implementation of the bond featurizer used in Chemprop.

    The bond features include:
    * A zero if the bond is not None. This seems really useless, but we follow the ref implementation
    * **One hot encoding of the bond type**. The supported bond types are
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.


    See Also
    --------
    BaseBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """

    def __init__(self, bond_data_field='e'):
        super(ChempropBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [lambda bond: [0],
                 bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 partial(bond_stereo_one_hot, encode_unknown=True)
                 # encode_unknown seems unnecessary as one of the options is STEREONONE. But we still follow the ref implementation.
                 ]
            )}, self_loop=False)


class RDKitMorganFingerprinter:
    """
    Molecule featurization with Morgan fingerprint
    """

    def __init__(self, radius=3, n_bits=1024):
        self.radius = radius
        self.n_bits = n_bits

    def process(self, smiles):
        mol = MolFromSmiles(smiles)
        fp = GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        arr = np.zeros(self.n_bits)
        ConvertToNumpyArray(fp, arr)
        return arr

    @property
    def feat_size(self) -> int:
        return self.n_bits


class RDKit2DGlobalFeaturizer:
    """
    Molecule featurization with RDKit 2D features. Uses descriptastorus (https://github.com/bp-kelley/descriptastorus).
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize (bool): If True, normalize feature values. The normalization uses a CDF fitted on a NIBR compound
                                catalogue. Default True.
        """
        if normalize:
            self.features_generator = MakeGenerator(("rdkit2dnormalized",))
        else:
            self.features_generator = MakeGenerator(("rdkit2d",))

    def process(self, smiles: str) -> list:
        features = self.features_generator.process(smiles)
        if features is None:  # fail
            raise ValueError(f"ERROR: could not generate rdkit features for SMILES '{smiles}'")
        else:
            return features[1:]  # do not return the initial 'True'.

    @property
    def feat_size(self) -> int:
        return len(self.features_generator.process("C")) - 1  # -1 for the initial 'True' that we do not return


def dummy_atom_featurizer(m: Mol):
    """For testing. Featurizes every atom with its index"""
    feats = [[a.GetIdx()] for a in m.GetAtoms()]
    return {"x": torch.FloatTensor(feats)}


def dummy_bond_featurizer(m: Mol):
    """For testing. Featurizes every bond with its index"""
    feats = [[b.GetIdx()] for b in m.GetBonds() for _ in range(2)]
    return {"e": torch.FloatTensor(feats)}
