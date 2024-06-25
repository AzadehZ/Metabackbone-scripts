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


def find_cross_over_in_longest_strand(dna):
    
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
    closest_pairs = similar_distance_pairs_sorted[:2] # Get up to two closest pairs
    
    
    crossover_positions = [cross_over_bases_max[0].pos, cross_over_bases_max[1].pos]
    closest_pairs_positions = [(pair[0].pos, pair[1].pos) for pair in closest_pairs]
    return cross_over_bases_max, max_index_difference, min_distance, closest_pairs, crossover_positions, closest_pairs_positions


def calculate_position(dna, left_indices, right_indices, t):
    
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
    
    P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
    return P



def find_valid_P(dna, left_indices, right_indices, cross_over_bases_max, closest_pairs_positions, sphere_radius):
    
    t_values = np.linspace(0, 1, 100)
    np.random.shuffle(t_values)
    
    crossover_positions = [cross_over_bases_max[0].pos, cross_over_bases_max[1].pos]
    tolerance_distance = sphere_radius / 2  # Adjust as needed

    for t in t_values:
        P = calculate_position(dna, left_indices, right_indices, t)
        far_enough = all(
            np.linalg.norm(P - np.array(pos)) > tolerance_distance
            for pos in crossover_positions
        )
        far_enough_pairs = all(
            np.linalg.norm(P - np.array(pair_pos[0])) > tolerance_distance and
            np.linalg.norm(P - np.array(pair_pos[1])) > tolerance_distance
            for pair_pos in closest_pairs_positions
        )

        if far_enough and far_enough_pairs:
            return P, t
    
    raise ValueError("Could not find a suitable P far enough from the crossover positions.")


def find_bases_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere = []
    base_to_strand_mapping = {}
    
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - P)

            if distance < sphere_radius:
                bases_in_sphere.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
    
    return bases_in_sphere, base_to_strand_mapping


def remove_one_strand_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, P, sphere_radius)
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


def remove_two_strands_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, P, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    dna_structures = []
    strand_pairs = [(strand_1, strand_2) for i, strand_1 in enumerate(strands_to_remove) for strand_2 in list(strands_to_remove)[i + 1:]]
    
    for strand_1, strand_2 in strand_pairs:
        strand_list = []
        for idx, strand in enumerate(dna.strands):
            if idx not in {strand_1, strand_2}:
                strand_list.append(strand)
        
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        dna_structures.append(new_dna_structure)
    return dna_structures


def remove_three_strands_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, P, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
      
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
    
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
        
        # Store the information about removed strands
        removed_strands_info.append((strand_1, strand_2, strand_3))
        print(f"Removed strands: {strand_1}, {strand_2}, {strand_3}")
        
    return dna_structures, removed_strands_info



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


def find_cross_over_in_longest_strand(dna):
    
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
    
    
    crossover_positions = [cross_over_bases_max[0].pos, cross_over_bases_max[1].pos]
    closest_pairs_positions = [(pair[0].pos, pair[1].pos) for pair in closest_pairs]
    
    return cross_over_bases_max, max_index_difference, min_distance, closest_pairs, crossover_positions, closest_pairs_positions

def calculate_position(dna, left_indices, right_indices, t):
    
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
    
    P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
    return P

def find_valid_P(dna, left_indices, right_indices, cross_over_bases_max, closest_pairs_positions, sphere_radius):
    t_values = np.linspace(0, 1, 100)
    np.random.shuffle(t_values)
    
    crossover_positions = [cross_over_bases_max[0].pos, cross_over_bases_max[1].pos]
    tolerance_distance = sphere_radius / 3.5

    for t in t_values:
        P = calculate_position(dna, left_indices, right_indices, t)
        far_enough = all(
            np.linalg.norm(P - np.array(pos)) > tolerance_distance
            for pos in crossover_positions
        )
        far_enough_pairs = all(
            np.linalg.norm(P - np.array(pair_pos[0])) > tolerance_distance and
            np.linalg.norm(P - np.array(pair_pos[1])) > tolerance_distance
            for pair_pos in closest_pairs_positions
        )

        if far_enough and far_enough_pairs:
            return P, t
    
    raise ValueError("Could not find a suitable P far enough from the crossover positions.")

def find_bases_in_sphere(dna, P, sphere_radius):
    bases_in_sphere = []
    base_to_strand_mapping = {}
    
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - P)
            # print(f"Base UID: {base.uid}, Position: {base_position}, Distance to P: {distance}")

            if distance < sphere_radius:
                bases_in_sphere.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
                # print(f"Base UID {base.uid} is within the sphere radius")
    
    return bases_in_sphere, base_to_strand_mapping


def remove_one_strand_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, P, sphere_radius)
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


def remove_two_strands_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, P, sphere_radius)
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
   

    dna_structures = []
    removed_strands_info = []

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

def remove_three_strands_in_sphere(dna, P, sphere_radius):
    
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, P, sphere_radius)
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
        print(f"Removed strands: {strand_1}, {strand_2}, {strand_3}")
        
    return dna_structures, removed_strands_info


eq_steps = 1e5
prod_steps = 1e5
rel_steps = 1e3
    
# Parameters for simulations
eq_parameters = {'dt':f'0.003','steps':f'{eq_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                 'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}

prod_parameters = {'dt':f'0.003','steps':f'{prod_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                   'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}
rel_parameters = {'steps': f'{rel_steps}', 'max_backbone_force': '200', 'max_backbone_force_far': '200'}

sim_base_path = '/home/ava/Dropbox (ASU)/temp/Metabackbone/metabackbone/notebooks/Simulations/simulations/Automatically_rmvd_staples/one_staple_removed'
base_path = '/home/ava/Dropbox (ASU)/temp/Metabackbone/structure_files/six_helix_oxdna_file/Automatically_removed_staples/1512_bp/one_staple_remvd'


# Function to queue relaxation simulations for all structures
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

    # Process all queued relaxation simulations
    simulation_manager.worker_manager(gpu_mem_block=False)
    print("Completed all relaxation simulations")
    return sim_list_rel

# Function to queue equilibration simulations for all structures
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

    # Process all queued equilibration simulations
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

    simulation_manager.worker_manager(gpu_mem_block=False)
    
    return sim_list_prod


def run_all_simulations(structures, base_path, sim_base_path):
    print("Starting relaxation simulations...")
    sim_list_rel = queue_relaxation_simulations(structures, base_path, sim_base_path)

    print("Starting equilibration simulations...")
    sim_list_eq = queue_equilibration_simulations(structures, base_path, sim_base_path)

    print("Starting production simulations...")
    sim_list_prod = queue_production_simulations(structures, base_path, sim_base_path)
    
    return sim_list_rel, sim_list_eq, sim_list_prod