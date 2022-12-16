from typing import List, Union, Optional

from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts, SanitizeRxn


class SLAPReactionGenerator:
    def __init__(self):
        # TODO verify these reaction drafts
        piperazine_rxn = ReactionFromSmarts(
            "[#6]-[#6]-[#8]-[#6](=O)-[#7]-1-[#6]-[#6H1](-[#6:1])-[#7]-[#6H1](-[#6:2])-[#6]-1.>>[#6:1]-[#6]=O.[#6:2]-[#6]=O"
        )
        dimemorpholine_rxn = ReactionFromSmarts(
            "[#6:1]-[#6]-1-[#6]-[#8][#6]([#6])([#6])[#6](-[#6:2])-[#7]-1>>[#6:1]-[#6]=O.[#6:2]-[#6]=O"
        )
        monomemorpholine_rxn = ReactionFromSmarts(
            "[#6:1]-[#6]-1-[#6]-[#8]-[#6](-[#6])-[#6](-[#6:2])-[#7]-1>>[#6:1]-[#6]=O.[#6:2]-[#6]=O"
        )
        trialphamorpholine_rxn = ReactionFromSmarts(
            "[#6:4]-[#6]-1-[#6]-[#8]-[#6]-[#6D4:3]-[#7]-1>>[#6:3]=O.[#6:4]-[#6]=O"
        )
        morpholine_rxn = ReactionFromSmarts(
            "[#6:1]-[#6]-1-[#6]-[#8]-[#6]-[#6](-[#6:2])-[#7]-1>>[#6:1]-[#6]=O.[#6:2]-[#6]=O"
        )

        self.reactions = {
            "piperazine": piperazine_rxn,
            "dimemorpholine": dimemorpholine_rxn,
            "monomemorpholine": monomemorpholine_rxn,
            "trialphamorpholine": trialphamorpholine_rxn,
            "morpholine": morpholine_rxn,
        }

        for rxn in self.reactions.values():
            rxn.Initialize()
            SanitizeRxn(rxn)

    def generate_reactions(
        self, product: Union[str, Chem.Mol], starting_materials: tuple = ()
    ) -> List[ChemicalReaction]:
        """
        Takes a SLAP reaction product and generates the possible reactions leading to this product.

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
            starting_materials(tuple, optional): Allowed starting materials. If given, reactions will only be returned if
                all generated starting materials are contained in this tuple.

        Returns:
            list: A list of the atom-mapped reactions leading to the specified product using the SLAP platform.
        """
        if isinstance(product, str):
            product_mol = Chem.MolFromSmiles(product)
        else:
            product_mol = Chem.Mol(product)

        # we try applying to the reaction templates from the more to the less stringent. We stop as soon as a template works

        # piperazine
        # TODO put in the entire enumeration / atommapping shenenigans
