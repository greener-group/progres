# Progres - Protein Graph Embedding Search

This repository contains the method in the pre-print:

...

It provides the `progres` Python package that lets you search structures against pre-embedded structural databases and pre-embed datasets for searching against. Searching typically takes 1-2 s and is faster for multiple queries. Currently [SCOPe](https://scop.berkeley.edu), [CATH](http://cathdb.info) and [ECOD](http://prodata.swmed.edu/ecod) are provided for searching against but we aim to add others such as the [AlphaFold database](https://alphafold.ebi.ac.uk) in future.

This is work in progress software - the implementation, API and results may change.
Please open issues or [contact me](http://jgreener64.github.io) with any feedback.
Appropriate version numbers will distinguish versions.
Training scripts and datasets will be made available on publication.

## Installation

1. Python 3.6 or later is required. The software is OS-independent.
2. Install [PyTorch](https://pytorch.org) 1.11 or later, [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) as appropriate for your system. A GPU is not required and will only provide speedup in certain situations since multiple workers can be used on CPU. Example commands:
```bash
conda install pytorch==1.12.0 -c pytorch
conda install pytorch-scatter pyg -c pyg
```

3. Run `pip install progres`, which will also install [Biopython](https://biopython.org) and [mmtf-python](https://github.com/rcsb/mmtf-python) if they are not already present.
4. The first time you run the software the trained model and pre-embedded databases (~x MB) will be downloaded to the package directory, which requires an internet connection.

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
# PARAMETERS: minsimilarity 0.8, maxhits 100, progres v0.1.0
# HIT_N  DOMAIN   HIT_NRES  SIMILARITY  NOTES
      1  d1a6ja_       150      1.0000  d.112.1.1 - Nitrogen regulatory bacterial protein IIa-ntr {Escherichia coli [TaxId: 562]}
      2  d3urra1       151      0.9992  d.112.1.0 - automated matches {Burkholderia thailandensis [TaxId: 271848]}
      3  d2a0ja_       146      0.9982  d.112.1.0 - automated matches {Neisseria meningitidis [TaxId: 122586]}
      4  d2oqta1       150      0.9976  d.112.1.0 - automated matches {Streptococcus pyogenes [TaxId: 301447]}
      5  d1xiza1       153      0.9970  d.112.1.1 - Putative PTS protein STM3784 {Salmonella typhimurium [TaxId: 90371]}
...
```
- `-q` is the path to the query structure file. Alternatively, `-l` is a text file with one query file path per line and each result will be printed in turn. This is considerably faster for multiple queries since setup only occurs once and multiple workers can be used.
- `-t` is the pre-embedded database to search against. Currently this must be either one of the databases listed below or the file path to a pre-embedded dataset generated with `progres embed`.
- `-f` determines the file format of the query structure (`guess`, `pdb`, `mmcif`, `mmtf` or `coords`). By default this is guessed from the file extension, with `pdb` chosen if a guess can't be made. `coords` refers to a text file with the coordinates of a Cα atom separated by white space on each line.
- `-s` is the minimum similarity threshold above which to return hits, default 0.8.
- `-m` is the maximum number of hits to return, default 100.

Query structures should be a single protein domain, though it can be discontinuous (chain IDs are ignored).
You can slice out domains manually using software such as the `pdb_selres` command from [pdb-tools](http://www.bonvinlab.org/pdb-tools).

The available pre-embedded databases are:

| Name      | Description                                                                           | Number of domains | Search time (1 query) | Search time (100 queries) |
| --------- | ------------------------------------------------------------------------------------- | ----------------- | --------------------- | ------------------------- |
| `scope95` | ASTRAL set of [SCOPe](https://scop.berkeley.edu) 2.08 domains clustered at 95% seq ID | 35,371            | 1.49 s                | 3.29 s                    |
| `scope40` | ASTRAL set of [SCOPe](https://scop.berkeley.edu) 2.08 domains clustered at 40% seq ID | 15,127            | 1.46 s                | 2.96 s                    |
| `cath40`  | S40 non-redundant domains from [CATH](http://cathdb.info) 23/11/22                    | 31,884            | 1.52 s                | 3.25 s                    |
| `ecod70`  | F70 representative domains from [ECOD](http://prodata.swmed.edu/ecod) develop287      | x                 | x                     | x                         |

Search time is for a 150 residue protein (d1a6ja_ in PDB format) on an Intel i9-10980XE CPU with PyTorch 1.11.
Times are given for 1 or 100 queries.

## Pre-embed a dataset to search against

To embed a dataset of structures, allowing it to be searched against:
```bash
progres embed -l filepaths.txt -o searchdb.pt
```
- `-l` is a text file with information on one structure per line, each of which will be one entry in the output. White space should separate the file path to the structure and the domain name, with optionally any additional text being treated as a note for the notes column of the results.
- `-o` is the output file path for the PyTorch file containing a dictionary with the embeddings and associated data. It can be read in with `torch.load`.
- `-f` determines the file format of each structure as above (`guess`, `pdb`, `mmcif`, `mmtf` or `coords`).

Again, the structures should correspond to single protein domains.

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
embs = torch.nn.functional.normalize(torch.rand(1000, 128), dim=1)
scores = pg.embedding_similarity(embs.unsqueeze(0), embs.unsqueeze(1))
scores.shape # torch.Size([1000, 1000])
```

## Notes

The implementation of the E(n)-equivariant GNN uses [EGNN PyTorch](https://github.com/lucidrains/egnn-pytorch).
