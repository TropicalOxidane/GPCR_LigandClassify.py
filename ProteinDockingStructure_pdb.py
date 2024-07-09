import requests

pdb_id = input("")

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
