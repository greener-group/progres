Scripts for benchmarking other methods.

Set up the `pdbstyle-2.08` directory of PDB files with something like:
```bash
wget https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz
tar -xvf pdbstyle-sel-gs-bib-40-2.08.tgz
cd pdbstyle-2.08
mv */*.ent .
rmdir ??
cd ..
mv pdbstyle-2.08 pdbstyle-2.08_models
mkdir pdbstyle-2.08
julia extract_model.jl
rm -r pdbstyle-2.08_models pdbstyle-sel-gs-bib-40-2.08.tgz
```
