import os
from typing import List, Union, Tuple, Sequence, Optional, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdChemReactions import (
    ChemicalReaction,
    ReactionFromSmarts,
    SanitizeRxn,
    ReactionToSmarts,
)

from src.util.rdkit_util import (
    create_reaction_instance,
    remove_mapno,
    canonicalize_smiles,
)


class SLAPReactionGenerator:
    _piperazine_rxn = "[#6]-[#6]-[#8]-[#6](=O)-[#7]-1-[#6]-[#6H1](-[#6:1])-[#7]-[#6H1](-[#6:2])-[#6]-1.>>[#6:1]-[#6]=O.[#6:2]-[#6]=O"
    _dimemorpholine_rxn = "[#6:1]-[#6]-1-[#6]-[#8]-[#6](-[#6])(-[#6])-[#6](-[#6:2])-[#7]-1>>[#6:2]-[#6]=O.[#6:1]-[#6]=O"
    _monomemorpholine_rxn = "[#6:1]-[#6]-1-[#6]-[#8]-[#6](-[#6])-[#6](-[#6:2])-[#7]-1>>[#6:2]-[#6]=O.[#6:1]-[#6]=O"
    _trialphamorpholine_rxn = (
        "[#6:4]-[#6]-1-[#6]-[#8]-[#6]-[#6D4:3]-[#7]-1>>[#6:3]=O.[#6:4]-[#6]=O"
    )
    _morpholine_rxn = (
        "[#6:1]-[#6]-1-[#6]-[#8]-[#6]-[#6](-[#6:2])-[#7]-1>>[#6:1]-[#6]=O.[#6:2]-[#6]=O"
    )
    _piperazine_slap_rxn = "[#6H1:1]=O>>[#6]-[#6]-[#8]-[#6](=O)-[#7](-[#6]-[#6:1]-[#7])-[#6]-[#14](-[#6])(-[#6])-[#6]"
    _dimemorpholine_slap_rxn = (
        "[#6H1:1]=O>>[#6]-[#6](-[#6])(-[#6:1]-[#7])-[#8]-[#6]-[#14](-[#6])(-[#6])-[#6]"
    )
    _monomemorpholine_slap_rxn = (
        "[#6H1:1]=O>>[#6]-[#6](-[#6:1]-[#7])-[#8]-[#6]-[#14](-[#6])(-[#6])-[#6]"
    )
    _morpholine_slap_rxn = (
        "[#6H1:1]=O>>[#6]-[#14](-[#6])(-[#6])-[#6]-[#8]-[#6]-[#6:1]-[#7]"
    )
    _trialphamorpholine_slap_rxn = "[#6:1]-[#6](-[#6:2])=O>>[#6]-[#14](-[#6])(-[#6])-[#6]-[#8]-[#6]-[#6](-[#6:1])(-[#6:2])-[#7]"
    _slap_rxn = "[#6:1]-[#6X3;H1,H0:2]=[#8].[#6:3]-[#6:4](-[#7:5])-[#6:6]-[#8,#7:7]-[#6:8]-[#14]>>[#6:1]-[#6:2]-1-[#6:8]-[#8,#7:7]-[#6:6]-[#6:4](-[#6:3])-[#7:5]-1"
    dataset = None

    def __init__(self):
        self.backwards_reactions = {
            "piperazine": ReactionFromSmarts(self._piperazine_rxn),
            "dimemorpholine": ReactionFromSmarts(self._dimemorpholine_rxn),
            "monomemorpholine": ReactionFromSmarts(self._monomemorpholine_rxn),
            "trialphamorpholine": ReactionFromSmarts(self._trialphamorpholine_rxn),
            "morpholine": ReactionFromSmarts(self._morpholine_rxn),
        }

        self.slap_forming_reactions = {
            "piperazine": ReactionFromSmarts(self._piperazine_slap_rxn),
            "dimemorpholine": ReactionFromSmarts(self._dimemorpholine_slap_rxn),
            "monomemorpholine": ReactionFromSmarts(self._monomemorpholine_slap_rxn),
            "trialphamorpholine": ReactionFromSmarts(self._trialphamorpholine_slap_rxn),
            "morpholine": ReactionFromSmarts(self._morpholine_slap_rxn),
        }

        self.slap_rxn = ReactionFromSmarts(self._slap_rxn)

        for rxn in self.backwards_reactions.values():
            rxn.Initialize()
            SanitizeRxn(rxn)
        for rxn in self.slap_forming_reactions.values():
            rxn.Initialize()
            SanitizeRxn(rxn)
        self.slap_rxn.Initialize()
        SanitizeRxn(self.slap_rxn)

    def _try_reaction(self, product_type, product_mol):
        reactants = self.backwards_reactions[product_type].RunReactants((product_mol,))
        if len(reactants) > 0:
            # sanitize before returning
            [[Chem.SanitizeMol(m) for m in pair] for pair in reactants]
            return reactants, product_type
        return None, None

    def generate_reactants(
        self, product: Union[str, Chem.Mol], starting_materials: tuple = ()
    ) -> Tuple[Tuple[Tuple[Chem.Mol]], str]:
        """
        Takes a SLAP reaction product and generates the possible reactants leading to this product.
        The aldehyde/ketone reactants are generated, not the SLAP reagents. The first aldehyde/ketone of each reactant
        pair is the one used to produce the SLAP reagent.

        Rules:
            - The product has to be a morpholine or mono-N-protected piperazine.
            - Piperazines have one substituent on each of the two carbons alpha to the free amine
            - Morpholines can have one substituent on each of the two carbons alpha to the free amine. One additional alpha-
                substituent is possible OR 1 OR 2 methyl groups on one of the beta carbons are possible
            - For piperazines, two reactions are generated
            - For beta-substituted (w.r.t. N) morpholines, one reaction is generated and the substituted side will be from
                the SLAP reagent.
            - For 3,3,5-trisubstituted morpholines, one reaction is generated and the disubstituted side will be from
                the SLAP reagent.
            - For 3,5-disubstituted morpholines, two reactions will be generated.

        Args:
            product (str or Chem.Mol): Product of a SLAP reaction.
            starting_materials(tuple, optional): Allowed starting materials. If given, starting materials will only be
                returned if both generated starting materials are contained in this tuple. Otherwise, this pair will not
                be returned.

        Returns:
            list: A list of reactants (as Mol objects) leading to the specified product using the SLAP platform.
        """
        if isinstance(product, str):
            product_mol = Chem.MolFromSmiles(product)
        else:
            product_mol = Chem.Mol(product)

        # we try applying to the reaction templates from the more to the less stringent. We stop as soon as a template works

        # piperazine
        reactants, product_type = self._try_reaction("piperazine", product_mol)
        if not reactants:
            # dimemorpholine
            reactants, product_type = self._try_reaction("dimemorpholine", product_mol)
        if not reactants:
            # monomemorpholine
            reactants, product_type = self._try_reaction(
                "monomemorpholine", product_mol
            )
        if not reactants:
            # trialphamorpholine
            reactants, product_type = self._try_reaction(
                "trialphamorpholine", product_mol
            )
        if not reactants:
            # morpholine
            reactants, product_type = self._try_reaction("morpholine", product_mol)
        if not reactants:
            raise ValueError("No reaction found for product.")

        # sanity check: there should never be more than two possible reactions
        if len(reactants) > 2:
            raise RuntimeError("More than two possible reactions found.")

        # check if the starting materials are allowed
        if starting_materials:
            invalid_reactants = []
            for i, reactant_pair in enumerate(reactants):
                for reactant in reactant_pair:
                    if not all([Chem.MolToSmiles(reactant) in starting_materials]):
                        invalid_reactants.append(i)
            reactants = (
                reactant
                for i, reactant in enumerate(reactants)
                if i not in invalid_reactants
            )

        # check if the reactions are distinct. If not, remove the second one
        if len(reactants) > 1:
            if Chem.MolToSmiles(reactants[0][0]) == Chem.MolToSmiles(
                reactants[1][0]
            ) and Chem.MolToSmiles(reactants[0][1]) == Chem.MolToSmiles(
                reactants[1][1]
            ):
                reactants = (reactants[0],)

        return reactants, product_type

    def generate_slap_reagent(self, reactant: Chem.Mol, product_type: str) -> Chem.Mol:
        """
        Takes a reactant and generates the corresponding SLAP reagent.

        Args:
            reactant (Chem.Mol): Reactant of a SLAP reaction.
            product_type (str): Type of the product.

        Returns:
            Chem.Mol: The SLAP reagent.
        """
        slap_reagent = self.slap_forming_reactions[product_type].RunReactants(
            (reactant,)
        )[0][0]
        Chem.SanitizeMol(slap_reagent)
        return slap_reagent

    def generate_reaction(
        self, reactant_pair: Sequence[Chem.Mol], product_type: str
    ) -> ChemicalReaction:
        """
        Generates an atom-mapped, unbalanced reaction from the reactants and product type.

        Args:
            reactant_pair (Sequence): Reactants of the SLAP reaction. Expects two aldehydes/ketones.
                The first item in the sequence is used to generate the SLAP reagent. The second item must be an aldehyde.
            product_type (str): Type of the product. Can be either "morpholine" or "piperazine" or "dimemorpholine" or "monomemorpholine" or "trialphamorpholine".

        Returns:
            ChemicalReaction: A ChemicalReaction object representing the SLAP reaction.
        """

        slap = self.generate_slap_reagent(reactant_pair[0], product_type)
        reaction = create_reaction_instance(
            self.slap_rxn, (reactant_pair[1], slap)
        )  # note that we give the slap reagent last because this is the way we defined the reaction.
        if len(reaction) > 1:
            raise RuntimeError("More than one reaction found.")
        elif len(reaction) == 0:
            raise RuntimeError("No reaction found.")
        return reaction[0]

    def generate_reactions_for_product(
        self,
        product: Union[str, Chem.Mol],
        starting_materials: tuple = (),
        return_additional_info: bool = False,
        return_strings: bool = False,
    ) -> Union[
        List[ChemicalReaction],
        Tuple[List[ChemicalReaction], List[List[Chem.Mol]], List[str]],
    ]:
        """
        Generates all possible SLAP reactions for a given product.

        Args:
            product (str or Chem.Mol): Product of a SLAP reaction.
            starting_materials(tuple, optional): Allowed starting materials. If given, a reaction will only be
                returned if both generated starting materials are contained in this tuple. Otherwise, this reaction will not
                be returned. Defaults to False.
            return_additional_info (bool): Whether to additionally return the reactants and product type.
            return_strings (bool): Whether to return the reactions as reactionSMILES strings (as opposed to RDKit
                objects). Defaults to False.

        Returns:
            list: A list of ChemicalReaction objects representing the SLAP reactions leading to the given product.
        """
        reactants, product_type = self.generate_reactants(product, starting_materials)
        reactions = []
        for reactant_pair in reactants:
            reactions.append(self.generate_reaction(reactant_pair, product_type))
        if return_strings:
            reactions = [ReactionToSmarts(reaction) for reaction in reactions]

        if return_additional_info:
            return reactions, reactants, [product_type for _ in reactions]
        else:
            return reactions

    def reactants_in_dataset(
        self,
        reactant_pair: Sequence[Chem.Mol],
        product_type: str,
        dataset_path: Optional[Union[str, os.PathLike]] = None,
        use_cache=True,
    ) -> Tuple[bool, bool, bool, Optional[list]]:
        """
        Check whether the reactants appear in a reference data set.

        Args:
            dataset_path (str): Path to the dataset. Expects a CSV file with the columns "SMILES"
            and "targets".
            reactant_pair (Sequence): Reactants of the SLAP reaction. Expects two aldehydes/ketones.
                The first item in the sequence is used to generate the SLAP reagent. The second item must be an aldehyde.
            product_type (str): Type of the product. Can be either "morpholine" or "piperazine" or "dimemorpholine" or "monomemorpholine" or "trialphamorpholine".


        Returns:
            tuple: The first item indicates whether the first reactant appears in the dataset. The second item indicates
                whether the second reactant appears in the dataset. The third item indicated whether this exact combination appears in the data set.
                If the third item is True, the forth item is a list of the
                reaction outcomes of the respective reactions in the dataset. Otherwise, it is None.
        """
        if use_cache:
            if not self.dataset:
                data = []
                reactions = pd.read_csv(
                    dataset_path, usecols=["SMILES", "targets"]
                ).values
                for reaction in reactions:
                    rxn = ReactionFromSmarts(reaction[0])
                    reactants = [
                        Chem.MolToSmiles(remove_mapno(rxn.GetReactantTemplate(0))),
                        Chem.MolToSmiles(remove_mapno(rxn.GetReactantTemplate(1))),
                    ]
                    reactants = [
                        canonicalize_smiles(smi, remove_explicit_H=True)
                        for smi in reactants
                    ]
                    outcome = reaction[1]
                    reactants.append(outcome)
                    data.append(reactants)
                self.dataset = data
            dataset = self.dataset
        else:
            data = []
            reactions = pd.read_csv(dataset_path, usecols=["SMILES", "targets"]).values
            for reaction in reactions:
                rxn = ReactionFromSmarts(reaction[0])
                reactants = [
                    Chem.MolToSmiles(remove_mapno(rxn.GetReactantTemplate(0))),
                    Chem.MolToSmiles(remove_mapno(rxn.GetReactantTemplate(1))),
                ]
                reactants = [
                    canonicalize_smiles(smi, remove_explicit_H=True)
                    for smi in reactants
                ]
                outcome = reaction[1]
                reactants.append(outcome)
                data.append(reactants)
            dataset = data

        slap_smiles = Chem.MolToSmiles(
            self.generate_slap_reagent(reactant_pair[0], product_type)
        )
        aldehyde = Chem.MolToSmiles(reactant_pair[1])
        dataset_flattened = [item for sublist in dataset for item in sublist]
        dataset_reactants = ["+".join(sublist[:2]) for sublist in dataset]
        slap_in_dataset = slap_smiles in dataset_flattened
        aldehyde_in_dataset = aldehyde in dataset_flattened
        reaction_outcomes = None
        if slap_in_dataset and aldehyde_in_dataset:
            # check whether the exact combination appears in the dataset
            reaction_idx_in_dataset = [
                i
                for i, reactants in enumerate(dataset_reactants)
                if (reactants == "+".join([slap_smiles, aldehyde]))
                or (reactants == "+".join([aldehyde, slap_smiles]))
            ]
            reaction_in_dataset = len(reaction_idx_in_dataset) > 0
            if reaction_in_dataset:
                reaction_outcomes = [dataset[i][2] for i in reaction_idx_in_dataset]
        else:
            reaction_in_dataset = False

        return (
            slap_in_dataset,
            aldehyde_in_dataset,
            reaction_in_dataset,
            reaction_outcomes,
        )
