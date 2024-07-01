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

def find_nearest_pairs(longest_strand, cross_over_bases_max, min_distance):
    
    num_bases = len(longest_strand)
    similar_distance_pairs = []
    distance_tolerance = 0.1 * min_distance
    vicinity_tolerance = 1.0  

    for i in range(num_bases):
        
        for j in range(i + 1, num_bases):
            
            base_i = longest_strand[i]
            base_j = longest_strand[j]
            distance = np.linalg.norm(np.array(base_i.pos) - np.array(base_j.pos))
            
            
            if abs(distance - min_distance) <= distance_tolerance:
                
                dist_to_max_0 = np.linalg.norm(np.array(base_i.pos) - np.array(cross_over_bases_max[0].pos))
                dist_to_max_1 = np.linalg.norm(np.array(base_j.pos) - np.array(cross_over_bases_max[1].pos))

                if dist_to_max_0 <= vicinity_tolerance and dist_to_max_1 <= vicinity_tolerance:
                    
                    if {base_i.uid, base_j.uid} != {cross_over_bases_max[0].uid, cross_over_bases_max[1].uid}:
                        similar_distance_pairs.append((base_i, base_j, distance))
    
    similar_distance_pairs_sorted = sorted(similar_distance_pairs, key=lambda x: abs(x[2] - min_distance))
    closest_pairs = similar_distance_pairs_sorted[:2]  

    return closest_pairs


def measure_pair_distances(closest_pairs):
    
    if len(closest_pairs) < 2:
        print("\nNot enough pairs in the closest_pairs list to perform the operation.\n")
        return

    first_pair = closest_pairs[0]
    second_pair = closest_pairs[1]

    base1_first_pair = first_pair[0]
    base2_first_pair = first_pair[1]
    base1_second_pair = second_pair[0]
    base2_second_pair = second_pair[1]

    distances = []

    def calculate_distance(base_a, base_b, description):
        distance = np.linalg.norm(np.array(base_a.pos) - np.array(base_b.pos))
        distances.append((description, base_a, base_b, distance))
        return distance

    distance_between_bases_1_2 = calculate_distance(base1_first_pair, base2_second_pair, "Pairing base1 of the first pair with base2 of the second pair")
    distance_between_bases_2_1 = calculate_distance(base2_first_pair, base1_second_pair, "Pairing base2 of the first pair with base1 of the second pair")
    distance_between_bases_2_2 = calculate_distance(base2_first_pair, base2_second_pair, "Pairing base2 of the first pair with base2 of the second pair")
    distance_between_base1s = calculate_distance(base1_first_pair, base1_second_pair, "Pairing base1 of the first pair with base1 of the second pair")

    for description, base_a, base_b, distance in distances:
        print(f"\n{description}:\n")
        print(f"Base A: UID = {base_a.uid}, Position = {base_a.pos}")
        print(f"Base B: UID = {base_b.uid}, Position = {base_b.pos}")
        print(f"Distance between them: {distance}\n")
        
        

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

    return cms_left_side, cms_right_side


def is_point_far_from_crossovers(point, crossover_positions, min_distance_threshold):
    
    for pos in crossover_positions:
        distance = np.linalg.norm(np.array(point) - np.array(pos))
        if distance < min_distance_threshold:
            return False
    return True


def find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold=5.0):
    
    cms_left_side, cms_right_side = calculate_position(dna, left_indices, right_indices)

    
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
        
def find_nearby_points(dna, point, min_distance, max_distance, num_points=5):
    
    left_nearby_points = []
    right_nearby_points = []

    left_indices_positions = [np.array(base.pos) for strand in dna.strands for base in strand if base.uid in left_indices]
    right_indices_positions = [np.array(base.pos) for strand in dna.strands for base in strand if base.uid in right_indices]
    
    cms_left_side = np.mean(left_indices_positions, axis=0)
    cms_right_side = np.mean(right_indices_positions, axis=0)
    
    direction_vector = cms_right_side - cms_left_side
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the direction vector

    for strand in dna.strands:
        for base in strand:
            pos = np.array(base.pos)
            dist = np.linalg.norm(pos - point)

            if min_distance < dist < max_distance:
                vector_from_P = pos - point
                projection = np.dot(vector_from_P, direction_vector)

                if projection < 0:  # Point is to the left of P along the reference axis
                    left_nearby_points.append(pos)
                else:  # Point is to the right of P along the reference axis
                    right_nearby_points.append(pos)

    if len(left_nearby_points) > num_points:
        indices = np.random.choice(range(len(left_nearby_points)), num_points, replace=False)
        left_nearby_points = [left_nearby_points[i] for i in indices]

    if len(right_nearby_points) > num_points:
        indices = np.random.choice(range(len(right_nearby_points)), num_points, replace=False)
        right_nearby_points = [right_nearby_points[i] for i in indices]

    return left_nearby_points, right_nearby_points

def calculate_angle_between_vectors(dna, left_indices, right_indices, point):
    
    left_points = []
    right_points = []

    for strand in dna.strands:
        for base in strand:
            if base.uid in left_indices:
                left_points.append(base.pos)
            elif base.uid in right_indices:
                right_points.append(base.pos)

    left_com = np.mean(left_points, axis=0)
    right_com = np.mean(right_points, axis=0)
    
    vector_left = left_com - point
    vector_right = right_com - point

    cos_theta = np.dot(vector_left, vector_right) / (np.linalg.norm(vector_left) * np.linalg.norm(vector_right))
    angle = np.arccos(cos_theta)

    return np.degrees(angle)

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


def remove_one_strand_in_sphere(dna, point, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}

    dna_structures = []
    
    for strand_index in strands_to_remove:
        strand_list = []
        
        for idx, strand in enumerate(dna.strands):
            if idx != strand_index:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
    
    return dna_structures

def remove_two_strands_in_sphere(dna, point, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)

    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}

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
    
    return dna_structures, removed_strands_info


def remove_three_strands_in_sphere(dna, point, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)

    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}

    dna_structures = []
    removed_strands_info = []


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
        
    return dna_structures, removed_strands_info



def export_dna_structures(new_dna_structures, base_path):
    """
    Exports new DNA structures to unique subdirectories and collects their output paths.
    
    Parameters:
    new_dna_structures : List[DNAStructure]
        List of new DNA structure objects to be saved.
    base_path : str
        The base directory path where the structures will be saved.
    
    Returns:
    List[dict]
        List of dictionaries containing structure ID and file paths.
    """
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


def run_all_simulations(structures, base_path, sim_base_path):
    
    print("Starting relaxation simulations...")
    sim_list_rel = queue_relaxation_simulations(structures, base_path, sim_base_path)

    print("Starting equilibration simulations...")
    sim_list_eq = queue_equilibration_simulations(structures, base_path, sim_base_path)

    print("Starting production simulations...")
    sim_list_prod = queue_production_simulations(structures, base_path, sim_base_path)
    
    return sim_list_rel, sim_list_eq, sim_list_prod
