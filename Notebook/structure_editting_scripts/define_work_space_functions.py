from itertools import combinations
import numpy as np
import sys
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf, from_path
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox
from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, TrajInfo
from pathlib import Path
import os
from ipy_oxdna.dna_structure import DNAStructure, DNAStructureStrand, load_dna_structure, DNABase, strand_from_info
from copy import deepcopy
from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager
import copy
from tqdm.auto import tqdm

from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager
import os

def load_dna_structure_files(base_path):
    
    dat_path = os.path.join(base_path, '1512_bp.dat')
    top_path = os.path.join(base_path, '1512_bp.top')
    dna = load_dna_structure(top_path, dat_path)
    return dna

def find_longest_strand(dna):
    longest_strand = None
    longest_strand_index = -1
    max_length = 0
    
    for index, strand in enumerate(dna.strands):
        if len(strand.bases) > max_length:
            max_length = len(strand.bases)
            longest_strand = strand
            longest_strand_index = index
    
    return longest_strand, longest_strand_index

def find_cross_over_in_longest_strand(longest_strand):
    
    min_distance = float('inf')
    max_index_difference = 0
    cross_over_bases_max = (None, None)
    num_bases = len(longest_strand)

    # Iterate over all pairs of bases in the DNA strand
    for i in range(num_bases):
        
        for j in range(i + 1, num_bases):
            
            base_i = longest_strand[i]
            base_j = longest_strand[j]
            index_difference = abs(base_i.uid - base_j.uid)
            distance = np.linalg.norm(np.array(base_i.pos) - np.array(base_j.pos))

            # Update if the index difference is the largest seen so far
            if index_difference > max_index_difference or (index_difference == max_index_difference and distance < min_distance):
                max_index_difference = index_difference
                min_distance = distance
                cross_over_bases_max = (base_i, base_j)

    return cross_over_bases_max, max_index_difference, min_distance

def calculate_position(dna, left_indices, right_indices):
    left_pos = []
    right_pos = []

    for strand in dna.strands:
        for base in strand:
            if base.uid in left_indices:
                left_pos.append(base.pos)

    for strand in dna.strands:
        for base in strand:
            if base.uid in right_indices:
                right_pos.append(base.pos)

    all_uids = [base.uid for strand in dna.strands for base in strand]

    if left_pos:
        cms_left_side = np.mean(left_pos, axis=0)
    else:
        raise ValueError("No positions found for left indices.")

    if right_pos:
        cms_right_side = np.mean(right_pos, axis=0)
    else:
        raise ValueError("No positions found for right indices.")

    t = random.uniform(0, 1)
    P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
    return P
