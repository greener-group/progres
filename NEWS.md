# Progres release notes

## v0.2.5 - Aug 2024

- Structures can now be split into domains with Chainsaw before searching, with each domain searched separately. This makes Progres suitable for use with multi-domain structures.
- The whole PDB split into domains with Chainsaw is made available to search against.
- Hetero atoms are now ignored during file reading.
- Example files are added for searching and database embedding.

## v0.2.4 - Jul 2024

- The `score` mode is added to calculate the Progres score between two structures.

## v0.2.3 - May 2024

- Incomplete downloads are handled during setup.

## v0.2.2 - Apr 2024

- The environmental variable `PROGRES_DATA_DIR` can be used to change where the downloaded data is stored.
- A Docker file is added.
- Searching on GPU is made more memory efficient.
- Bugs when running on Windows are fixed.

## v0.2.1 - Apr 2024

- The AlphaFold database TED domains are made available to search against, with FAISS used for fast searching.
- Pre-embedded databases are stored as Float16 to reduce disk usage.
- Datasets and scripts for benchmarking (including for other methods), FAISS index generation and training are made available.

## v0.2.0 - Mar 2023

- Change model architecture to use 6 EGNN layers and tau torsion angles, making it faster and SE(3)-invariant rather than E(3)-invariant.
- The AlphaFold models for 21 model organisms are made available to search against.
- The trained model and pre-embedded databases are downloaded from Zenodo rather than GitHub when first running the software.

## v0.1.3 - Nov 2022

- Fix data download.

## v0.1.2 - Nov 2022

- Add ECOD database.
- Use versioned model directory.

## v0.1.1 - Nov 2022

- Add einops dependency.
- Add code for ECOD database.

## v0.1.0 - Nov 2022

Initial release of the `progres` Python package for fast protein structure searching using structure graph embeddings.
