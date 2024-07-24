# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:31:02 2024

@author: gavjo
"""

import requests 
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# SMILES string for compound 
SMILES = input("SMILES: ")
pdb_id = input("pdb_id: ")

# Add explicit hydrogens, and generate molecule from SMILES
mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.UFFOptimizeMolecule(mol)

# Save the 3D coordinates to a PDB file
SMILES_file_path = "{SMILES}.pdb"
Chem.MolToPDBFile(mol, SMILES_file_path)

print("SMILES:" + SMILES)
print(f"3D structure saved to {SMILES_file_path}")

# URL to fetch the PDB file
pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

# Fetch the PDB file
r = requests.get(pdb_url)

# Check if the request was successful
if 200 <= r.status_code <= 299:
    pdb_file_path = f"{pdb_id}.pdb" 
    with open(pdb_file_path, 'w') as file:
        file.write(r.text)
    print(f"{pdb_id} saved as {pdb_file_path}")
else:
    print(f"Failed to download: {pdb_id}")
