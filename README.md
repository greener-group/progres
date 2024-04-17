# Progres - Protein Graph Embedding Search

[![Build status](https://github.com/greener-group/progres/workflows/CI/badge.svg)](https://github.com/greener-group/progres/actions)

This repository contains the method from the pre-print:

- Greener JG and Jamali K. Fast protein structure searching using structure graph embeddings. bioRxiv (2022) - [link](https://www.biorxiv.org/content/10.1101/2022.11.28.518224)

It provides the `progres` Python package that lets you search structures against pre-embedded structural databases and pre-embed datasets for searching against.
Searching typically takes 1-2 s and is much faster for multiple queries.
For the AlphaFold database, initial data loading takes around a minute but subsequent searching takes a tenth of a second per query.
Currently [SCOPe](https://scop.berkeley.edu), [CATH](http://cathdb.info), [ECOD](http://prodata.swmed.edu/ecod), the [AlphaFold structures for 21 model organisms](https://doi.org/10.1093/nar/gkab1061) and the [AlphaFold database TED domains](https://www.biorxiv.org/content/10.1101/2024.03.18.585509) are provided for searching against.

## Installation

1. Python 3.8 or later is required. The software is OS-independent.
2. Install [PyTorch](https://pytorch.org) 1.11 or later, [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [FAISS](https://github.com/facebookresearch/faiss) as appropriate for your system. A GPU is not required and will only provide speedup in certain situations since multiple workers can be used on CPU. Example commands:
```bash
conda create -n prog python=3.9
conda activate prog
conda install pytorch=1.11 faiss-cpu -c pytorch
conda install pytorch-scatter pyg -c pyg
```
3. Run `pip install progres`, which will also install [Biopython](https://biopython.org), [mmtf-python](https://github.com/rcsb/mmtf-python) and [einops](https://github.com/arogozhnikov/einops) if they are not already present.
4. The first time you search with the software the trained model and pre-embedded databases (~220 MB) will be downloaded to the package directory from [Zenodo](https://zenodo.org/record/7782088), which requires an internet connection. This can take a few minutes.
5. The first time you search against the AlphaFold database TED domains the pre-embedded database (~33 GB) will be downloaded in a similar way. This can take a while. Make sure you have enough disk space!

## Usage

On Unix systems the executable `progres` will be added to the path during installation.
On Windows you can call the `bin/progres` script with python if you can't access the executable.

Run `progres -h` to see the help text and `progres {mode} -h` to see the help text for each mode.
The modes are described below but there are other options outlined in the help text.
For example the `-d` flag sets the device to run on; this is `cpu` by default since this is usually fastest for searching, but `cuda` may be slightly faster when embedding a dataset.

## Searching a structure against a database

To search a PDB file `query.pdb` against domains in the SCOPe database and print output:
```bash
progres search -q query.pdb -t scope95
```
```
# QUERY_NUM: 1
# QUERY: query.pdb
# QUERY_SIZE: 150 residues
# DATABASE: scope95
# PARAMETERS: minsimilarity 0.8, maxhits 100, progres v0.2.1
# HIT_N  DOMAIN   HIT_NRES  SIMILARITY  NOTES
      1  d1a6ja_       150      1.0000  d.112.1.1 - Nitrogen regulatory bacterial protein IIa-ntr {Escherichia coli [TaxId: 562]}
      2  d2a0ja_       146      0.9988  d.112.1.0 - automated matches {Neisseria meningitidis [TaxId: 122586]}
      3  d3urra1       151      0.9983  d.112.1.0 - automated matches {Burkholderia thailandensis [TaxId: 271848]}
      4  d3lf6a_       154      0.9971  d.112.1.1 - automated matches {Artificial gene [TaxId: 32630]}
      5  d3oxpa1       147      0.9968  d.112.1.0 - automated matches {Yersinia pestis [TaxId: 214092]}
...
```
- `-q` is the path to the query structure file. Alternatively, `-l` is a text file with one query file path per line and each result will be printed in turn. This is considerably faster for multiple queries since setup only occurs once and multiple workers can be used.
- `-t` is the pre-embedded database to search against. Currently this must be either one of the databases listed below or the file path to a pre-embedded dataset generated with `progres embed`.
- `-f` determines the file format of the query structure (`guess`, `pdb`, `mmcif`, `mmtf` or `coords`). By default this is guessed from the file extension, with `pdb` chosen if a guess can't be made. `coords` refers to a text file with the coordinates of a Cα atom separated by white space on each line.
- `-s` is the minimum similarity threshold above which to return hits, default 0.8. As discussed in the paper, 0.8 indicates the same fold.
- `-m` is the maximum number of hits to return, default 100.

Query structures should be a single protein domain, though it can be discontinuous (chain IDs are ignored).
Tools such as [Merizo](https://github.com/psipred/Merizo), [SWORD2](https://www.dsimb.inserm.fr/SWORD2) and [Chainsaw](https://github.com/JudeWells/chainsaw) can be used to predict domains from a larger structure.
You can also slice out domains manually using software such as the `pdb_selres` command from [pdb-tools](http://www.bonvinlab.org/pdb-tools).

Interpreting the hit descriptions depends on the database being searched.
The domain name often includes a reference to the corresponding PDB file, for example d1a6ja_ refers to PDB ID 1A6J chain A, and this can be opened in the [RCSB PDB structure view](https://www.rcsb.org/3d-view/1A6J/1) to get a quick look.
For the AlphaFold database TED domains, files can be downloaded from [links such as this](https://alphafold.ebi.ac.uk/files/AF-A0A6J8EXE6-F1-model_v4.pdb) where `AF-A0A6J8EXE6-F1` is the first part of the hit notes and is followed by the residue range of the domain.

The available pre-embedded databases are:

| Name      | Description                                                                                                                                                                                | Number of domains | Search time (1 query)      | Search time (100 queries)  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | -------------------------- | -------------------------- |
| `scope95` | ASTRAL set of [SCOPe](https://scop.berkeley.edu) 2.08 domains clustered at 95% seq ID                                                                                                      | 35,371            | 1.35 s                     | 2.81 s                     |
| `scope40` | ASTRAL set of [SCOPe](https://scop.berkeley.edu) 2.08 domains clustered at 40% seq ID                                                                                                      | 15,127            | 1.32 s                     | 2.36 s                     |
| `cath40`  | S40 non-redundant domains from [CATH](http://cathdb.info) 23/11/22                                                                                                                         | 31,884            | 1.38 s                     | 2.79 s                     |
| `ecod70`  | F70 representative domains from [ECOD](http://prodata.swmed.edu/ecod) develop287                                                                                                           | 71,635            | 1.46 s                     | 3.82 s                     |
| `af21org` | [AlphaFold](https://alphafold.ebi.ac.uk) structures for 21 model organisms split into domains by [CATH-Assign](https://doi.org/10.1038/s42003-023-04488-9)                                 | 338,258           | 2.21 s                     | 11.0 s                     |
| `afted`   | [AlphaFold database](https://alphafold.ebi.ac.uk) structures split into domains by [TED](https://www.biorxiv.org/content/10.1101/2024.03.18.585509) and clustered at 50% sequence identity | 53,344,209        | 67.7 s                     | 73.1 s                     |

Search time is for a 150 residue protein (d1a6ja_ in PDB format) on an Intel i9-10980XE CPU with 256 GB RAM and PyTorch 1.11.
Times are given for 1 or 100 queries.
Note that `afted` uses exhaustive FAISS searching.
This doesn't change the hits that are found, but the similarity score will differ by a small amount - see the paper.

## Pre-embed a dataset to search against

To embed a dataset of structures, allowing it to be searched against:
```bash
progres embed -l filepaths.txt -o searchdb.pt
```
- `-l` is a text file with information on one structure per line, each of which will be one entry in the output. White space should separate the file path to the structure and the domain name, with optionally any additional text being treated as a note for the notes column of the results.
- `-o` is the output file path for the PyTorch file containing a dictionary with the embeddings and associated data. It can be read in with `torch.load`.
- `-f` determines the file format of each structure as above (`guess`, `pdb`, `mmcif`, `mmtf` or `coords`).

Again, the structures should correspond to single protein domains.
The embeddings are stored as Float16, which has no noticeable effect on search performance.

## Python library

`progres` can also be used in Python, allowing it to be integrated into other methods:
```python
import progres as pg

# Search as above, returns a list where each entry is a dictionary for a query
# A generator is also available as pg.progres_search_generator
results = pg.progres_search(querystructure="query.pdb", targetdb="scope95")
results[0].keys() # dict_keys(['query_num', 'query', 'query_size', 'database', 'minsimilarity',
                  #            'maxhits', 'domains', 'hits_nres', 'similarities', 'notes'])

# Pre-embed as above, saves a dictionary
pg.progres_embed(structurelist="filepaths.txt", outputfile="searchdb.pt")
import torch
torch.load("searchdb.pt").keys() # dict_keys(['ids', 'embeddings', 'nres', 'notes'])

# Read a structure file into a PyTorch Geometric graph
graph = pg.read_graph("query.pdb")
graph # Data(x=[150, 67], edge_index=[2, 2758], coords=[150, 3])

# Embed a single structure
embedding = pg.embed_structure("query.pdb")
embedding.shape # torch.Size([128])

# Load and reuse the model for speed
model = pg.load_trained_model()
embedding = pg.embed_structure("query.pdb", model=model)

# Embed Cα coordinates and search with the embedding
# This is useful for using progres in existing pipelines that give out Cα coordinates
# queryembeddings should have shape (128) or (n, 128)
#   and should be normalised across the 128 dimension
coords = pg.read_coords("query.pdb")
embedding = pg.embed_coords(coords) # Can take a list of coords or a tensor of shape (nres, 3)
results = pg.progres_search(queryembeddings=embedding, targetdb="scope95")

# Get the similarity score (0 to 1) between two embeddings
# The distance (1 - similarity) is also available as pg.embedding_distance
score = pg.embedding_similarity(embedding, embedding)
score # tensor(1.) in this case since they are the same embedding

# Get all-v-all similarity scores between 1000 embeddings
embs = torch.nn.functional.normalize(torch.randn(1000, 128), dim=1)
scores = pg.embedding_similarity(embs.unsqueeze(0), embs.unsqueeze(1))
scores.shape # torch.Size([1000, 1000])
```

## Scripts

Datasets and scripts for benchmarking (including for other methods), FAISS index generation and training are in the `scripts` directory.
The trained model and pre-embedded databases are available on [Zenodo](https://zenodo.org/record/7782088).

## Notes

The implementation of the E(n)-equivariant GNN uses [EGNN PyTorch](https://github.com/lucidrains/egnn-pytorch).

Please open issues or [get in touch](http://jgreener64.github.io) with any feedback.
