# Extract first model from ASTRAL PDB files

using BioStructures

in_dir  = "pdbstyle-2.08_models"
out_dir = "pdbstyle-2.08"
known_problems = ["d6a1ia1.ent"]

fs = readdir(in_dir)

for (fi, f) in enumerate(fs)
    println(fi, " / ", length(fs))
    f in known_problems && continue
    out_fp = joinpath(out_dir, f)
    isfile(out_fp) && continue
    s = read(joinpath(in_dir, f), PDBFormat)
    writepdb(out_fp, s[1])
end
