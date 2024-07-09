import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# SMILES string for compound 
SMILES = input("")

# Add explicit hydrogens, and generate molecule from SMILES
mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.UFFOptimizeMolecule(mol)

# Save the 3D coordinates to a PDB file
pdb_file_path = "{SMILES}.pdb"
Chem.MolToPDBFile(mol, pdb_file_path)

print("SMILES:" + SMILES)
print(f"3D structure saved to {pdb_file_path}")
