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
import random


colors = {'blue': '34', 'green': '32', 'yellow': '33', 'cyan': '36', 'red': '31', 'magenta': '35'}

def print_colored(message, color_code):
    print(f"\033[{color_code}m{message}\033[0m")

# Loading DNA structure files
def load_dna_structure_files(input_path):
    dat_path = os.path.join(input_path, '1512_bp.dat')
    top_path = os.path.join(input_path, '1512_bp.top')
    dna = load_dna_structure(top_path, dat_path)
    return dna

# Finding the longest strand in the DNA structure
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

# Finding cross-over in the longest strand
def find_cross_over_in_longest_strand(longest_strand):
    min_distance = float('inf')
    max_index_difference = 0
    cross_over_bases_max = (None, None)
    num_bases = len(longest_strand)
    for i in range(num_bases):
        for j in range(i + 1, num_bases):
            base_i = longest_strand[i]
            base_j = longest_strand[j]
            index_difference = abs(base_i.uid - base_j.uid)
            distance = np.linalg.norm(np.array(base_i.pos) - np.array(base_j.pos))
            if index_difference > max_index_difference or (index_difference == max_index_difference and distance < min_distance):
                max_index_difference = index_difference
                min_distance = distance
                cross_over_bases_max = (base_i, base_j)
    return cross_over_bases_max, max_index_difference, min_distance

# Calculating positions
def calculate_left_right_pos(dna, left_indices, right_indices):
    left_pos = []
    right_pos = []
    
    # print_colored(f"Left indices: {left_indices}", colors['blue'])
    # print_colored(f"Right indices: {right_indices}", colors['blue'])
    
    for strand in dna.strands:
        for base in strand:
            # print_colored(f"Base UID: {base.uid}", colors['blue'])  # Debug statement
            if base.uid in left_indices:
                left_pos.append(base.pos)
            elif base.uid in right_indices:
                right_pos.append(base.pos)
                
    if left_pos:
        cms_left_side = np.mean(left_pos, axis=0)
    else:
        raise ValueError("No positions found for left indices.")
    
    if right_pos:
        cms_right_side = np.mean(right_pos, axis=0)
    else:
        raise ValueError("No positions found for right indices.")
    
    # print_colored(f"Center of mass for the left side: {cms_left_side}", colors['green'])
    # print_colored(f"Center of mass for the right side: {cms_right_side}", colors['green'])
    
    midpoint = (cms_left_side + cms_right_side) / 2
    # print_colored(f"Midpoint between the left and right sides: {midpoint}", colors['green'])
    
    return cms_left_side, cms_right_side, midpoint

# Check if point is far from crossovers
def is_point_far_from_crossovers(point, crossover_positions, min_distance_threshold):
    for pos in crossover_positions:
        distance = np.linalg.norm(np.array(point) - np.array(pos))
        if distance < min_distance_threshold:
            return False
    return True

# Finding valid point
def find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold = 2.5):
    cms_left_side, cms_right_side, midpoint = calculate_left_right_pos(dna, left_indices, right_indices)
    
    cross_over_bases, max_index_difference, min_distance = find_cross_over_in_longest_strand(longest_strand)
    crossover_positions = [base.pos for base in cross_over_bases if base is not None]
    
    t = random.uniform(0, 1)
    first_P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
    if not crossover_positions:
        return first_P
    
    while True:
        t = random.uniform(0, 1)
        P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
        if is_point_far_from_crossovers(P, crossover_positions, min_distance_threshold):
            return P

# Finding bases around a point
def find_bases_around_point(dna, point, min_distance, max_distance):
    left_bases = []
    right_bases = []
    left_base_indices = []
    left_strand_indices = []
    right_base_indices = []
    right_strand_indices = []
    
    for strand_index, strand in enumerate(dna.strands):
        for base_index, base in enumerate(strand):
            distance = np.linalg.norm(np.array(base.pos) - np.array(point))
            if min_distance < distance < max_distance:
                if base.pos[0] < point[0]:
                    left_bases.append(base.pos)
                    left_base_indices.append(base_index)
                    left_strand_indices.append(strand_index)
                else:
                    right_bases.append(base.pos)
                    right_base_indices.append(base_index)
                    right_strand_indices.append(strand_index)
                    
    if left_bases:
        cms_left_bases = np.mean(left_bases, axis=0)
        # print(f"Center of mass for left bases around: {cms_left_bases}")
    else:
        cms_left_bases = None
        print("No left bases found.")
    
    if right_bases:
        cms_right_bases = np.mean(right_bases, axis=0)
        # print(f"Center of mass for right bases around: {cms_right_bases}")
    else:
        cms_right_bases = None
        print("No right bases found.")
    
    return (left_bases, right_bases, 
            cms_left_bases, cms_right_bases, 
            left_base_indices, right_base_indices, 
            left_strand_indices, right_strand_indices)

# Calculating center of mass
def calculate_center_of_mass(positions):
    if not positions:
        raise ValueError("No positions provided for center of mass calculation.")
    return np.mean(positions, axis=0)

# Calculating bend angle
def calculate_bend_angle(P, cms_left, cms_right):
    vec_left = cms_left - P
    vec_right = cms_right - P
    unit_vec_left = vec_left / np.linalg.norm(vec_left)
    unit_vec_right = vec_right / np.linalg.norm(vec_right)
    dot_product = np.dot(unit_vec_left, unit_vec_right)
    angle = np.arccos(dot_product) * (180.0 / np.pi)
    return angle

# Finding bend angle
def find_bend_angle(dna, left_indices, right_indices, longest_strand, point_pos, min_distance_threshold = 2.5, min_distance = 7.0, max_distance = 20.0):
    (left_bases, right_bases, 
    cms_left_bases, cms_right_bases, 
    left_base_indices, right_base_indices, 
    left_strand_indices, right_strand_indices) = find_bases_around_point(dna, point_pos, min_distance, max_distance)  
    cms_left = calculate_center_of_mass(left_bases)
    cms_right = calculate_center_of_mass(right_bases)
    bend_angle = calculate_bend_angle(point_pos, cms_left, cms_right)
    return bend_angle

# Calculating angles for all structures
def calculate_angles_for_all_structures(dna_list, left_indices, right_indices, min_distance_threshold = 2.5, min_distance = 7.0, max_distance = 20.0):
    angles = []
    for dna in dna_list:
        longest_strand, _ = find_longest_strand(dna)
        point_pos, bend_angle = find_bend_angle(dna, left_indices, right_indices, longest_strand, min_distance_threshold, min_distance, max_distance)
        angles.append((point_pos, bend_angle))
    return angles

# # Finding bases in sphere and determining lost staples
# def find_bases_in_sphere(reference_dna, mutant_dna_list, point, sphere_radius):
#     reference_bases, _ = find_bases_in_sphere(reference_dna, point, sphere_radius)
#     lost_staples = []
#     lost_staples_dict = {}

#     for mutant_index, mutant_dna in enumerate(mutant_dna_list):
#         mutant_bases, _ = find_bases_in_sphere(mutant_dna, point, sphere_radius)
#         lost_in_mutant = list(set(reference_bases) - set(mutant_bases))
#         lost_staples_dict[f'mutant_{mutant_index}'] = lost_in_mutant
#         lost_staples.extend(lost_in_mutant)
    
#     return lost_staples_dict, list(set(lost_staples))



# Finding bases in sphere
def find_bases_in_sphere(dna, point, sphere_radius):
    bases_in_sphere = []
    base_to_strand_mapping = {}
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - point)
            if distance < sphere_radius:
                bases_in_sphere.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
    return bases_in_sphere, base_to_strand_mapping



# Finding bases above point in sphere
def find_bases_above_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    above_bases = []
    base_to_strand_mapping = {}
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - point)
            if min_distance < distance < sphere_radius and base_position[1] > point[1]:
                above_bases.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
    return above_bases, base_to_strand_mapping

# Finding bases below point in sphere
def find_bases_below_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    below_bases = []
    base_to_strand_mapping = {}
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - point)
            if min_distance < distance < sphere_radius and base_position[1] < point[1]:
                below_bases.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
    return below_bases, base_to_strand_mapping


# Removing all the staples in the sphere
def remove_all_staples_in_sphere_except_longest(dna, point, sphere_radius):
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)

    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases in sphere:", bases_in_sphere)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    new_strands = [strand for idx, strand in enumerate(dna.strands) if idx not in strands_to_remove]
    new_dna_structure = DNAStructure(new_strands, dna.time, dna.box, dna.energy)
    
    print(f"Total number of strands removed: {len(strands_to_remove)}")
    return new_dna_structure, list(strands_to_remove)


# Removing one strand above point in sphere
def remove_one_strand_above_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_above, base_to_strand_mapping = find_bases_above_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    strands_above = list(set(base_to_strand_mapping.values()))
    print("Strand indices above the point and within the sphere:", strands_above)
    print("Strand indices to be removed:", list(strands_to_remove))
    
    dna_structures = []
    for strand_index in strands_to_remove:
        print(f"Removing strand index: {strand_index}")
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx != strand_index:
                strand_list.append(strand)
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures

# Removing one strand below point in sphere
def remove_one_strand_below_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_below, base_to_strand_mapping = find_bases_below_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    strands_below = list(set(base_to_strand_mapping.values()))
    print("Strand indices below the point and within the sphere:", strands_below)
    print("Strand indices to be removed:", list(strands_to_remove))
    
    dna_structures = []
    for strand_index in strands_to_remove:
        print(f"Removing strand index: {strand_index}")
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx != strand_index:
                strand_list.append(strand)
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures

# Removing one strand in sphere
def remove_one_strand_in_sphere(dna, point, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    strands_in_sphere = list(set(base_to_strand_mapping.values()))
    
    print("Strand indices in the sphere:", strands_in_sphere)
    print("Strand indices to be removed:", list(strands_to_remove))
    
    dna_structures = []
    removed_strands = []  # To store the strands that were removed

    for strand_index in strands_to_remove:
        print(f"Removing strand index: {strand_index}")
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx != strand_index:
                strand_list.append(strand)
            else:
                removed_strands.append(strand_index)  # Log the removed strand
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures, removed_strands


# Removing two strands above point in sphere
def remove_two_strands_above_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_above, base_to_strand_mapping = find_bases_above_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases above the point in sphere:", bases_above)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    dna_structures = []
    removed_strands_info = []

    # Create all possible pairs of strands to remove
    strand_pairs = [(strand_1, strand_2) for i, strand_1 in enumerate(strands_to_remove)
                    for strand_2 in list(strands_to_remove)[i + 1:]]

    for strand_1, strand_2 in strand_pairs:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
        
        removed_strands_info.append((strand_1, strand_2))
        print(f"Removed strands: {strand_1}, {strand_2}")
    
    return dna_structures, removed_strands_info

# Removing two strands below point in sphere
def remove_two_strands_below_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_below, base_to_strand_mapping = find_bases_below_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases below the point in sphere:", bases_below)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    dna_structures = []
    removed_strands_info = []

    # Create all possible pairs of strands to remove
    strand_pairs = [(strand_1, strand_2) for i, strand_1 in enumerate(strands_to_remove)
                    for strand_2 in list(strands_to_remove)[i + 1:]]

    for strand_1, strand_2 in strand_pairs:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
        
        removed_strands_info.append((strand_1, strand_2))
        print(f"Removed strands: {strand_1}, {strand_2}")
    
    return dna_structures, removed_strands_info

# Removing two strands in sphere
def remove_two_strands_in_sphere(dna, point, sphere_radius):
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases in sphere:", bases_in_sphere)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    dna_structures = []
    removed_strands_info = []

    # Create all possible pairs of strands to remove
    strand_pairs = [(strand_1, strand_2) for i, strand_1 in enumerate(strands_to_remove)
                    for strand_2 in list(strands_to_remove)[i + 1:]]

    for strand_1, strand_2 in strand_pairs:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
        
        removed_strands_info.append((strand_1, strand_2))
        print(f"Removed strands: {strand_1}, {strand_2}")
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures, removed_strands_info

# Removing three strands above point in sphere
def remove_three_strands_above_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_above, base_to_strand_mapping = find_bases_above_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases above the point in sphere:", bases_above)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    dna_structures = []
    removed_strands_info = []

    # Create all possible triplets of strands to remove
    strand_triplets = [(strand_1, strand_2, strand_3) for i, strand_1 in enumerate(strands_to_remove)
                       for j, strand_2 in enumerate(list(strands_to_remove)[i + 1:])
                       for strand_3 in list(strands_to_remove)[i + j + 2:]]

    for strand_1, strand_2, strand_3 in strand_triplets:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2, strand_3}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
        
        removed_strands_info.append((strand_1, strand_2, strand_3))
        print(f"Removed strands: {strand_1}, {strand_2}, {strand_3}")
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures, removed_strands_info

# Removing three strands below point in sphere
def remove_three_strands_below_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_below, base_to_strand_mapping = find_bases_below_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases below the point in sphere:", bases_below)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    dna_structures = []
    removed_strands_info = []

    # Create all possible triplets of strands to remove
    strand_triplets = [(strand_1, strand_2, strand_3) for i, strand_1 in enumerate(strands_to_remove)
                       for j, strand_2 in enumerate(list(strands_to_remove)[i + 1:])
                       for strand_3 in list(strands_to_remove)[i + j + 2:]]

    for strand_1, strand_2, strand_3 in strand_triplets:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2, strand_3}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
        
        removed_strands_info.append((strand_1, strand_2, strand_3))
        print(f"Removed strands: {strand_1}, {strand_2}, {strand_3}")
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures, removed_strands_info

# Removing three strands in sphere
def remove_three_strands_in_sphere(dna, point, sphere_radius):
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases in sphere:", bases_in_sphere)
    print("Base to strand mapping:", base_to_strand_mapping)
    # print("Longest strand:", longest_strand)
    # print("Longest strand index:", longest_strand_index)
    
    strands_in_sphere = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands in sphere to consider for removal:", strands_in_sphere)

    dna_structures = []
    removed_strands_info = []

    # Create all possible triplets of strands to remove
    strand_triplets = [(strand_1, strand_2, strand_3) for i, strand_1 in enumerate(strands_in_sphere)
                       for j, strand_2 in enumerate(list(strands_in_sphere)[i + 1:])
                       for strand_3 in list(strands_in_sphere)[i + j + 2:]]

    for strand_1, strand_2, strand_3 in strand_triplets:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2, strand_3}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
        
        removed_strands_info.append((strand_1, strand_2, strand_3))
        print(f"Removed strands: {strand_1}, {strand_2}, {strand_3}")
    
    print(f"Total number of new structures created: {len(dna_structures)}")
    return dna_structures, removed_strands_info

# Function to store removed strands and their bases
def stored_removed_strands(dna, removed_strands_info):
    summaries = []
    for i, (strand_1, strand_2, strand_3) in enumerate(removed_strands_info):
        strands = [strand_1, strand_2, strand_3]
        removed_bases = []
        for strand in strands:
            removed_bases.extend([base.uid for base in dna.strands[strand]])
        summary = f"For structure_{i}, staples with the indexes {strand_1}, {strand_2}, {strand_3} were removed."
        summaries.append((summary, removed_bases))
    return summaries  


# Removing all staples above point in sphere
def remove_all_staples_above_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_above, base_to_strand_mapping = find_bases_above_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases above the point in sphere:", bases_above)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    new_strands = []
    for idx, strand in enumerate(dna.strands):
        if idx not in strands_to_remove:
            new_strands.append(strand)
    
    new_dna_structure = DNAStructure(new_strands, dna.time, dna.box, dna.energy)
    
    print(f"Total number of strands removed: {len(strands_to_remove)}")
    return new_dna_structure, list(strands_to_remove)

# Removing all staples below point in sphere
def remove_all_staples_below_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    bases_below, base_to_strand_mapping = find_bases_below_point_in_sphere(dna, point, sphere_radius, min_distance)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    print("Bases below the point in sphere:", bases_below)
    print("Base to strand mapping:", base_to_strand_mapping)
    print("Longest strand:", longest_strand)
    print("Longest strand index:", longest_strand_index)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    print("Strands to remove:", strands_to_remove)

    strand_list = []
    for idx, strand in enumerate(dna.strands):
        if idx not in strands_to_remove:
            strand_list.append(strand)
    
    new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
    
    print(f"Total number of strands removed: {len(strands_to_remove)}")
    return new_dna_structure, list(strands_to_remove)

# Exporting DNA structures
def export_dna_structures(new_dna_structures, base_path):
    output_paths = []
    for i, new_dna in enumerate(new_dna_structures):
        structure_id = i
        unique_subdir = os.path.join(base_path, f'structure_{structure_id}')
        os.makedirs(unique_subdir, exist_ok=True)
        dat_path = os.path.join(unique_subdir, '1512_bp_rmv_staples.dat')
        top_path = os.path.join(unique_subdir, '1512_bp_rmv_staples.top')
        new_dna.export_top_conf(Path(top_path), Path(dat_path))
        output_paths.append({
            'structure_id': structure_id,
            'dat_path': dat_path,
            'top_path': top_path
        })
    return output_paths

# Queueing relaxation simulations
def queue_relaxation_simulations(structures, base_path, sim_base_path):
    simulation_manager = SimulationManager()
    sim_list_rel = []
    for structure_id, structure in enumerate(structures):
        file_dir = os.path.join(base_path, f'structure_{structure_id}')
        sim_path = os.path.join(sim_base_path, f'1512_bp_{structure_id}')
        rel_dir = os.path.join(sim_path, 'relaxed')
        if not os.path.exists(file_dir):
            print(f"Directory does not exist: {file_dir}")
            continue
        sim_relax = Simulation(file_dir, rel_dir)
        sim_relax.build(clean_build='force')
        sim_relax.input.swap_default_input("cpu_MC_relax")
        sim_relax.input_file(rel_parameters)
        simulation_manager.queue_sim(sim_relax)
        sim_list_rel.append(sim_relax)
        print(f"Queued relaxation simulation for structure {structure_id}")
    simulation_manager.worker_manager(gpu_mem_block=False)
    print("Completed all relaxation simulations")
    return sim_list_rel

# Queueing equilibration simulations
def queue_equilibration_simulations(structures, base_path, sim_base_path):
    simulation_manager = SimulationManager()
    sim_list_eq = []
    for structure_id, structure in enumerate(structures):
        sim_path = os.path.join(sim_base_path, f'1512_bp_{structure_id}')
        rel_dir = os.path.join(sim_path, 'relaxed')
        eq_dir = os.path.join(sim_path, 'eq')
        if not os.path.exists(rel_dir):
            print(f"Directory does not exist: {rel_dir}")
            continue
        sim_eq = Simulation(rel_dir, eq_dir)
        sim_eq.build(clean_build='force')
        sim_eq.input_file(eq_parameters)
        simulation_manager.queue_sim(sim_eq)
        sim_list_eq.append(sim_eq)
        print(f"Queued equilibration simulation for structure {structure_id}")
    simulation_manager.worker_manager(gpu_mem_block=False)
    print("Completed all equilibration simulations")
    return sim_list_eq

# Queueing production simulations
def queue_production_simulations(structures, base_path, sim_base_path):
    simulation_manager = SimulationManager()
    sim_list_prod = []
    for structure_id, structure in enumerate(structures):
        sim_path = os.path.join(sim_base_path, f'1512_bp_{structure_id}')
        eq_dir = os.path.join(sim_path, 'eq')
        prod_dir = os.path.join(sim_path, 'prod')
        if not os.path.exists(eq_dir):
            print(f"Directory does not exist: {eq_dir}")
            continue
        sim_prod = Simulation(eq_dir, prod_dir)
        sim_prod.build(clean_build='force')
        sim_prod.input_file(prod_parameters)
        simulation_manager.queue_sim(sim_prod)
        sim_list_prod.append(sim_prod)
        print(f"Queued production simulation for structure {structure_id}")
    simulation_manager.worker_manager(gpu_mem_block=False)
    print("Completed all production simulations")
    return sim_list_prod

# Running all simulations
def run_all_simulations(structures, base_path, sim_base_path):
    print("Starting relaxation simulations...")
    sim_list_rel = queue_relaxation_simulations(structures, base_path, sim_base_path)
    print("Starting equilibration simulations...")
    sim_list_eq = queue_equilibration_simulations(structures, base_path, sim_base_path)
    print("Starting production simulations...")
    sim_list_prod = queue_production_simulations(structures, base_path, sim_base_path)
    return sim_list_rel, sim_list_eq, sim_list_prod

