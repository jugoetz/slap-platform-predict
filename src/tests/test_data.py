from unittest import TestCase

from rdkit import Chem
from rdkit.Chem.rdChemReactions import ReactionToSmarts

from src.data.util import SLAPReactionGenerator
from src.util.rdkit_util import canonicalize_smiles


class TestSLAPReactionGenerator(TestCase):
    def setUp(self):
        self.morpholine = "CC(C)C1COCC(N1)c1ccccc1"  # from isobutanal + benzaldehyde (either can be SLAP)
        self.piperazine = "CCOC(=O)N1CC(Cc2cccc(c2)C(F)(F)F)NC(C1)c1ccc2n(C)cnc2c1"  # from 2-(4-trifluoromethylphenyl)enthanal + 5-formyl-N-methylbenzimidazole (either can be SLAP)
        self.memorpholine = "CCCC1NC(COC1C)c1ccc(cc1)C(C)(C)C"  # from butanal (SLAP)  + 4 tert-butyl benzaldehyde
        self.dimemorpholine = "CC(C)(C)c1ccc(cc1)C1COC(C)(C)C(N1)c1ccc(Cl)cc1"  # from 4-chlorobenzaldehyde (SLAP) + 4 tert-butyl benzaldehyde
        self.trialphamorpholine = "CC(C)(C)c1ccc(cc1)C1COCC2(CCCCC2)N1"  # from cyclohexanone (SLAP) + 4 tert-butyl benzaldehyde
        self.isobutanal = "CC(C)C=O"
        self.isobutanal_morpholine_slap = canonicalize_smiles("CC(C)C(N)COC[Si](C)(C)C")
        self.isobutanal_monomemorpholine_slap = canonicalize_smiles(
            "CC(C)C(N)C(C)OC[Si](C)(C)C"
        )
        self.isobutanal_piperazine_slap = canonicalize_smiles(
            "CCOC(=O)N(CC(N)C(C)C)C[Si](C)(C)C"
        )
        self.benzaldehyde = "c1ccccc1C=O"
        self.benzaldehyde_morpholine_slap = canonicalize_smiles(
            "C[Si](C)(C)COCC(N)C1=CC=CC=C1"
        )
        self.benzaldehyde_dimemorpholine_slap = canonicalize_smiles(
            "CC(C)(OC[Si](C)(C)C)C(N)C1=CC=CC=C1"
        )
        self.cyclohexanone = "O=C1CCCCC1"
        self.cyclohexanone_spiromorpholine_slap = canonicalize_smiles(
            "C[Si](C)(C)COCC1(N)CCCCC1"
        )
        self.generator = SLAPReactionGenerator()

    def test_generate_reactants(self):
        """Test breaking products into the respective reactant pairs"""

        reactants, _ = self.generator.generate_reactants(self.morpholine)
        with self.subTest("Number of unique reactant pairs for morpholine"):
            self.assertEqual(len(reactants), 2)
        for reactant_pair in reactants:
            with self.subTest(
                "Number of reactants must be 2 per pair", reactants=reactant_pair
            ):
                self.assertEqual(len(reactant_pair), 2)

        reactants, _ = self.generator.generate_reactants(self.piperazine)
        with self.subTest("Number of unique reactant pairs for piperazine"):
            self.assertEqual(len(reactants), 2)
        for reactant_pair in reactants:
            with self.subTest(
                "Number of reactants must be 2 per pair", reactants=reactant_pair
            ):
                self.assertEqual(len(reactant_pair), 2)

        reactants, _ = self.generator.generate_reactants(self.memorpholine)
        with self.subTest("Number of unique reactant pairs for Me-morpholine"):
            self.assertEqual(len(reactants), 1)
        for reactant_pair in reactants:
            with self.subTest(
                "Number of reactants must be 2 per pair", reactants=reactant_pair
            ):
                self.assertEqual(len(reactant_pair), 2)

        reactants, _ = self.generator.generate_reactants(self.dimemorpholine)
        with self.subTest("Number of unique reactant pairs for Dime-morpholine"):
            self.assertEqual(len(reactants), 1)
        for reactant_pair in reactants:
            with self.subTest(
                "Number of reactants must be 2 per pair", reactants=reactant_pair
            ):
                self.assertEqual(len(reactant_pair), 2)

        reactants, _ = self.generator.generate_reactants(self.trialphamorpholine)
        with self.subTest("Number of unique reactant pairs for Trialphamorpholine"):
            self.assertEqual(len(reactants), 1)
        for reactant_pair in reactants:
            with self.subTest(
                "Number of reactants must be 2 per pair", reactants=reactant_pair
            ):
                self.assertEqual(len(reactant_pair), 2)

    def test_generate_slap_reagent(self):
        """Test generating a few example SLAP reagents."""
        with self.subTest("morpholine SLAP reagent for isobutanal"):
            slap = Chem.MolToSmiles(
                self.generator.generate_slap_reagent(
                    Chem.MolFromSmiles(self.isobutanal), "morpholine"
                )
            )
            self.assertEqual(slap, self.isobutanal_morpholine_slap)
        with self.subTest("morpholine SLAP reagent for benzaldehyde"):
            slap = Chem.MolToSmiles(
                self.generator.generate_slap_reagent(
                    Chem.MolFromSmiles(self.benzaldehyde), "morpholine"
                )
            )
            self.assertEqual(slap, self.benzaldehyde_morpholine_slap)
        with self.subTest("memorpholine SLAP reagent for isobutanal"):
            slap = Chem.MolToSmiles(
                self.generator.generate_slap_reagent(
                    Chem.MolFromSmiles(self.isobutanal), "monomemorpholine"
                )
            )
            self.assertEqual(slap, self.isobutanal_monomemorpholine_slap)
        with self.subTest("dimemorpholine SLAP reagent for benzaldehyde"):
            slap = Chem.MolToSmiles(
                self.generator.generate_slap_reagent(
                    Chem.MolFromSmiles(self.benzaldehyde), "dimemorpholine"
                )
            )
            self.assertEqual(slap, self.benzaldehyde_dimemorpholine_slap)
        with self.subTest("spiromorpholine SLAP reagent for cyclohexanone"):
            slap = Chem.MolToSmiles(
                self.generator.generate_slap_reagent(
                    Chem.MolFromSmiles(self.cyclohexanone), "trialphamorpholine"
                )
            )
            self.assertEqual(slap, self.cyclohexanone_spiromorpholine_slap)
        with self.subTest("piperazine SLAP reagent for isobutanal"):
            slap = Chem.MolToSmiles(
                self.generator.generate_slap_reagent(
                    Chem.MolFromSmiles(self.isobutanal), "piperazine"
                )
            )
            self.assertEqual(slap, self.isobutanal_piperazine_slap)

    def test_generate_reaction(self):
        with self.subTest("Reaction for isobutanal + benzaldehyde morpholine SLAP"):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.benzaldehyde),
                    Chem.MolFromSmiles(self.isobutanal),
                ),
                "morpholine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:9]-[#6:1](-[#6:10])-[#6:2]=[#8].[#6]-[Si](-[#6])(-[#6])-[#6:8]-[#8:7]-[#6:6]-[#6:4](-[#7:5])-[#6:3]1:[#6:11]:[#6:13]:[#6:15]:[#6:14]:[#6:12]:1>>[#6:1](-[#6:2]1-[#6:8]-[#8:7]-[#6:6]-[#6:4](-[#6:3]2:[#6:11]:[#6:13]:[#6:15]:[#6:14]:[#6:12]:2)-[#7:5]-1)(-[#6:9])-[#6:10]",
            )
        with self.subTest("Reaction for benzaldehyde + isobutanal morpholine SLAP"):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.isobutanal),
                    Chem.MolFromSmiles(self.benzaldehyde),
                ),
                "morpholine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:10]1:[#6:12]:[#6:13]:[#6:11]:[#6:9]:[#6:1]:1-[#6:2]=[#8].[#6]-[Si](-[#6])(-[#6])-[#6:8]-[#8:7]-[#6:6]-[#6:4](-[#7:5])-[#6:3](-[#6:14])-[#6:15]>>[#6:1]1(-[#6:2]2-[#6:8]-[#8:7]-[#6:6]-[#6:4](-[#6:3](-[#6:14])-[#6:15])-[#7:5]-2):[#6:9]:[#6:11]:[#6:13]:[#6:12]:[#6:10]:1",
            )
        with self.subTest(
            "Reaction for benzaldehyde + isobutanal monomemorpholine SLAP"
        ):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.isobutanal),
                    Chem.MolFromSmiles(self.benzaldehyde),
                ),
                "monomemorpholine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:10]1:[#6:12]:[#6:13]:[#6:11]:[#6:9]:[#6:1]:1-[#6:2]=[#8].[#6:16]-[#6:6](-[#6:4](-[#7:5])-[#6:3](-[#6:14])-[#6:15])-[#8:7]-[#6:8]-[Si](-[#6])(-[#6])-[#6]>>[#6:1]1(-[#6:2]2-[#6:8]-[#8:7]-[#6:6](-[#6:4](-[#6:3](-[#6:14])-[#6:15])-[#7:5]-2)-[#6:16]):[#6:9]:[#6:11]:[#6:13]:[#6:12]:[#6:10]:1",
            )
        with self.subTest("Reaction for benzaldehyde + isobutanal piperazine SLAP"):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.isobutanal),
                    Chem.MolFromSmiles(self.benzaldehyde),
                ),
                "piperazine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:10]1:[#6:12]:[#6:13]:[#6:11]:[#6:9]:[#6:1]:1-[#6:2]=[#8].[#6:20]-[#6:19]-[#8:17]-[#6:16](=[#8:18])-[#7:7](-[#6:6]-[#6:4](-[#7:5])-[#6:3](-[#6:14])-[#6:15])-[#6:8]-[Si](-[#6])(-[#6])-[#6]>>[#6:1]1(-[#6:2]2-[#6:8]-[#7:7](-[#6:6]-[#6:4](-[#6:3](-[#6:14])-[#6:15])-[#7:5]-2)-[#6:16](-[#8:17]-[#6:19]-[#6:20])=[#8:18]):[#6:9]:[#6:11]:[#6:13]:[#6:12]:[#6:10]:1",
            )
        with self.subTest("Reaction for isobutanal + benzaldehyde dimemorpholine SLAP"):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.benzaldehyde),
                    Chem.MolFromSmiles(self.isobutanal),
                ),
                "dimemorpholine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:9]-[#6:1](-[#6:10])-[#6:2]=[#8].[#6:16]-[#6:6](-[#6:17])(-[#6:4](-[#7:5])-[#6:3]1:[#6:11]:[#6:13]:[#6:15]:[#6:14]:[#6:12]:1)-[#8:7]-[#6:8]-[Si](-[#6])(-[#6])-[#6]>>[#6:1](-[#6:2]1-[#6:8]-[#8:7]-[#6:6](-[#6:4](-[#6:3]2:[#6:11]:[#6:13]:[#6:15]:[#6:14]:[#6:12]:2)-[#7:5]-1)(-[#6:16])-[#6:17])(-[#6:9])-[#6:10]",
            )
        with self.subTest(
            "Reaction for benzaldehyde + cyclohexanone spiromorpholine SLAP"
        ):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.cyclohexanone),
                    Chem.MolFromSmiles(self.benzaldehyde),
                ),
                "trialphamorpholine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:10]1:[#6:12]:[#6:13]:[#6:11]:[#6:9]:[#6:1]:1-[#6:2]=[#8].[#6]-[Si](-[#6])(-[#6])-[#6:8]-[#8:7]-[#6:6]-[#6:4]1(-[#6:3]-[#6:14]-[#6:15]-[#6:16]-[#6:17]-1)-[#7:5]>>[#6:1]1(-[#6:2]2-[#6:8]-[#8:7]-[#6:6]-[#6:4]3(-[#6:3]-[#6:14]-[#6:15]-[#6:16]-[#6:17]-3)-[#7:5]-2):[#6:9]:[#6:11]:[#6:13]:[#6:12]:[#6:10]:1",
            )
        with self.subTest(
            "Reaction for isobutanal + cyclohexanone spiromorpholine SLAP"
        ):
            reaction = self.generator.generate_reaction(
                (
                    Chem.MolFromSmiles(self.cyclohexanone),
                    Chem.MolFromSmiles(self.isobutanal),
                ),
                "trialphamorpholine",
            )
            self.assertEqual(
                ReactionToSmarts(reaction),
                "[#6:9]-[#6:1](-[#6:10])-[#6:2]=[#8].[#6]-[Si](-[#6])(-[#6])-[#6:8]-[#8:7]-[#6:6]-[#6:4]1(-[#6:3]-[#6:11]-[#6:12]-[#6:13]-[#6:14]-1)-[#7:5]>>[#6:1](-[#6:2]1-[#6:8]-[#8:7]-[#6:6]-[#6:4]2(-[#6:3]-[#6:11]-[#6:12]-[#6:13]-[#6:14]-2)-[#7:5]-1)(-[#6:9])-[#6:10]",
            )
