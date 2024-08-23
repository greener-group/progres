# !/usr/bin/env python
"""
Script to convert mmCIF files to PDB format.
usage: python cif2pdb.py ciffile [pdbfile]
Requires python BioPython (`pip install biopython`). It should work with recent version of python 2 or 3.
@author Spencer Bliven <spencer.bliven@gmail.com>
"""
import os
import sys
import argparse
import logging
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.mmtf import MMTFParser
from Bio.PDB import PDBIO

LOG = logging.getLogger(__name__)


def int_to_chain(i, base=62):
    """
    int_to_chain(int,int) -> str
    Converts a positive integer to a chain ID. Chain IDs include uppercase
    characters, numbers, and optionally lowercase letters.
    i = a positive integer to convert
    base = the alphabet size to include. Typically 36 or 62.
    """
    if i < 0:
        raise ValueError("positive integers only")
    if base < 0 or 62 < base:
        raise ValueError("Invalid base")

    quot = int(i) // base
    rem = i % base
    if rem < 26:
        letter = chr(ord("A") + rem)
    elif rem < 36:
        letter = str(rem - 26)
    else:
        letter = chr(ord("a") + rem - 36)
    if quot == 0:
        return letter
    else:
        return int_to_chain(quot - 1, base) + letter


class OutOfChainsError(Exception): 
    pass


def rename_chains(structure):
    """Renames chains to be one-letter chains

    Existing one-letter chains will be kept. Multi-letter chains will be truncated
    or renamed to the next available letter of the alphabet.

    If more than 62 chains are present in the structure, raises an OutOfChainsError

    Returns a map between new and old chain IDs, as well as modifying the input structure
    """
    next_chain = 0  #
    # single-letters stay the same
    chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}
    for o in structure.get_chains():
        if len(o.id) != 1:
            if o.id[0] not in chainmap:
                chainmap[o.id[0]] = o.id
                o.id = o.id[0]
            else:
                c = int_to_chain(next_chain)
                while c in chainmap:
                    next_chain += 1
                    c = int_to_chain(next_chain)
                    if next_chain >= 62:
                        raise OutOfChainsError()
                chainmap[c] = o.id
                o.id = c
    return chainmap

def cif2pdb(ciffile, pdbfile, fileformat):
    strucid = os.path.split(ciffile)[-1].split(".")[0]
    # Read file
    if fileformat == "mmcif":
        parser = MMCIFParser()
        structure = parser.get_structure(strucid, ciffile)
    elif fileformat == "mmtf":
        structure = MMTFParser.get_structure(ciffile)
    else:
        raise ValueError(f"Unrecognized file format: {fileformat}")
    # rename long chains
    try:
        chainmap = rename_chains(structure)
    except OutOfChainsError:
        logging.error("Too many chains to represent in PDB format")
        sys.exit(1)

    for new, old in chainmap.items():
        if new != old:
            logging.info("Renaming chain {0} to {1}".format(old, new))

    for atom in structure.get_atoms():
        if atom.get_serial_number() > 99990:
            atom.set_serial_number(99990) # Leave space for TER

    # Write PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile, preserve_atom_numbering=True)
