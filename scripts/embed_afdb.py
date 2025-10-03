# Embed TED-100 domains with progres
# This script is illustrative of the method
# The PyTorch output files are combined into one file afterwards

import progres as pg
import torch
import urllib.request
import os
import sys

run_n = int(sys.argv[1])

cluster_rep_file = f"cluster_rep_splits/split_{run_n}.txt"
out_file = f"cluster_rep_embs/split_{run_n}.pt"
temp_afdb_file = f"temp_pdbs/temp_afdb_{run_n}.pdb"
temp_dom_file = f"temp_pdbs/temp_dom_{run_n}.pdb"

embs, domids, nres, notes = [], [], [], []

with open(cluster_rep_file) as f:
    afdb_id_last = ""
    for line in f.readlines():
        ted_id, chopping, nres_dom, plddt, cath_label, tax = line.rstrip().split()
        afdb_id = ted_id.split("-model")[0]
        domain_res = []
        for res_range in chopping.split("_"):
            res_start, res_end = res_range.split("-")
            domain_res.extend(list(range(int(res_start), int(res_end) + 1)))
        domain_res = set(domain_res)

        # Previous domain may be from the same model
        if afdb_id != afdb_id_last:
            if os.path.isfile(temp_afdb_file):
                os.remove(temp_afdb_file)
            urllib.request.urlretrieve(
                f"https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v4.pdb",
                temp_afdb_file,
            )

        with open(temp_afdb_file) as f2, open(temp_dom_file, "w") as of:
            for line2 in f2.readlines():
                if line2.startswith("ATOM"):
                    resnum = int(line2[22:26])
                    if resnum in domain_res:
                        of.write(line2)

        cath_label = "N/A" if cath_label == "-" else cath_label
        tax        = "N/A" if tax        == "-" else tax
        assert "-" not in [afdb_id, chopping, plddt, cath_label, tax]

        domids.append(ted_id)
        nres.append(int(nres_dom))
        notes.append(f"{afdb_id} {chopping} - pLDDT {plddt} - {cath_label} - {tax}")
        emb = pg.embed_structure(temp_dom_file)
        embs.append(emb)
        afdb_id_last = afdb_id

torch.save(
    {
        "ids"       : domids,
        "embeddings": torch.stack(embs),
        "nres"      : nres,
        "notes"     : notes,
    },
    out_file,
)

if os.path.isfile(temp_afdb_file):
    os.remove(temp_afdb_file)
if os.path.isfile(temp_dom_file):
    os.remove(temp_dom_file)
