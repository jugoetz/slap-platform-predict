from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string.

    Removes any atom-mapping numbers
    """
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            a.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)


def move_atom_index_to_mapno(mol: Mol):
    """
    Write the atom indexes to property "molAtomMapNo", so they can be displayed in drawing.

    Note that this overwrites any previous "molAtomMapNo".
    """
    mol_copy = Chem.Mol(mol)  # necessary to not change the input molecule
    for i, a in enumerate(mol_copy.GetAtoms()):

        if a.HasProp("molAtomMapNumber"):
            a.ClearProp("molAtomMapNumber")
        a.SetProp("molAtomMapNumber", str(i))
    return mol_copy


def move_bond_index_to_props(mol: Mol):
    """
    Write the bond indexes to property "bondIndex", so they can be displayed in drawing.
    """
    mol_copy = Chem.Mol(mol)  # necessary to not change the input molecule
    for i, b in enumerate(mol_copy.GetBonds()):
        b.SetProp("bondNote", str(i))
    return mol_copy


def mol_to_file_with_indices(mol: Mol, file: str):
    """
    Draw a molecule to file with indices for atoms and bonds displayed
    """
    mol = move_atom_index_to_mapno(mol)
    mol = move_bond_index_to_props(mol)
    Draw.MolToFile(mol, file)
