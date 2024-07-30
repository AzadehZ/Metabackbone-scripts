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
import random

def load_dna_structure_files(input_path):
    dat_path = os.path.join(input_path, '1512_bp.dat')
    top_path = os.path.join(input_path, '1512_bp.top')
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
            if index_difference > max_index_difference or (index_difference == max_index_difference and distance < min_distance):
                max_index_difference = index_difference
                min_distance = distance
                cross_over_bases_max = (base_i, base_j)
    return cross_over_bases_max, max_index_difference, min_distance

def calculate_left_right_pos(dna, left_indices, right_indices):
    left_pos = []
    right_pos = []
    
    for strand in dna.strands:
        for base in strand:
            if base.uid in left_indices:
                left_pos.append(base.pos)
                print(f"Left index {base.uid} has position {base.pos}")
            elif base.uid in right_indices:
                right_pos.append(base.pos)
                print(f"Right index {base.uid} has position {base.pos}")
                
    if left_pos:
        cms_left_side = np.mean(left_pos, axis=0)
    else:
        raise ValueError("No positions found for left indices.")
    
    if right_pos:
        cms_right_side = np.mean(right_pos, axis=0)
    else:
        raise ValueError("No positions found for right indices.")
    
    print(f"Center of mass for the left side: {cms_left_side}")
    print(f"Center of mass for the right side: {cms_right_side}")
    
    midpoint = (cms_left_side + cms_right_side) / 2
    print(f"Midpoint between the left and right sides: {midpoint}")
    
    return cms_left_side, cms_right_side, midpoint

def is_point_far_from_crossovers(point, crossover_positions, min_distance_threshold):
    for pos in crossover_positions:
        distance = np.linalg.norm(np.array(point) - np.array(pos))
        if distance < min_distance_threshold:
            return False
    return True

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
                if base.pos[0] < point[0]:    # the x-coordinate of the base
                    left_bases.append(base.pos)
                    left_base_indices.append(base_index)
                    left_strand_indices.append(strand_index)
                else:
                    right_bases.append(base.pos)
                    right_base_indices.append(base_index)
                    right_strand_indices.append(strand_index)
                    
    if left_bases:
        cms_left_bases = np.mean(left_bases, axis=0)
        print(f"Center of mass for left bases around: {cms_left_bases}")
    else:
        cms_left_bases = None
        print("No left bases found.")
    
    if right_bases:
        cms_right_bases = np.mean(right_bases, axis=0)
        print(f"Center of mass for right bases around: {cms_right_bases}")
    else:
        cms_right_bases = None
        print("No right bases found.")
    
    return (left_bases, right_bases, 
            cms_left_bases, cms_right_bases, 
            left_base_indices, right_base_indices, 
            left_strand_indices, right_strand_indices)

def calculate_center_of_mass(positions):
    if not positions:
        raise ValueError("No positions provided for center of mass calculation.")
    return np.mean(positions, axis=0)

def calculate_bend_angle(P, cms_left, cms_right):
    vec_left = cms_left - P
    vec_right = cms_right - P
    unit_vec_left = vec_left / np.linalg.norm(vec_left)
    unit_vec_right = vec_right / np.linalg.norm(vec_right)
    dot_product = np.dot(unit_vec_left, unit_vec_right)
    angle = np.arccos(dot_product) * (180.0 / np.pi)
    return angle

def find_bend_angle(dna, left_indices, right_indices, longest_strand, min_distance_threshold = 2.5, min_distance = 7.0, max_distance = 20.0):
    point_pos = find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold)
    (left_bases, right_bases, 
    cms_left_bases, cms_right_bases, 
    left_base_indices, right_base_indices, 
    left_strand_indices, right_strand_indices) = find_bases_around_point(dna, point_pos, min_distance, max_distance)  
    cms_left = calculate_center_of_mass(left_bases)
    cms_right = calculate_center_of_mass(right_bases)
    bend_angle = calculate_bend_angle(point_pos, cms_left, cms_right)
    return point_pos, bend_angle

def calculate_angles_for_all_structures(dna_list, left_indices, right_indices, min_distance_threshold = 2.5, min_distance = 7.0, max_distance = 20.0):
    angles = []
    for dna in dna_list:
        longest_strand, _ = find_longest_strand(dna)
        point_pos, bend_angle = find_bend_angle(dna, left_indices, right_indices, longest_strand, min_distance_threshold, min_distance, max_distance)
        angles.append((point_pos, bend_angle))
    return angles

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

def find_bases_above_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    above_bases = []
    base_to_strand_mapping = {}
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - point)
            if min_distance < distance < sphere_radius and base_position[1] > point[1]:  # Above the point and within the sphere
                above_bases.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
    return above_bases, base_to_strand_mapping


def find_bases_below_point_in_sphere(dna, point, sphere_radius, min_distance=0):
    below_bases = []
    base_to_strand_mapping = {}
    for strand_index, strand in enumerate(dna.strands):
        for base in strand:
            base_position = np.array(base.pos)
            distance = np.linalg.norm(base_position - point)
            if min_distance < distance < sphere_radius and base_position[1] < point[1]:  # Below the point and within the sphere
                below_bases.append(base.uid)
                base_to_strand_mapping[base.uid] = strand_index
    return below_bases, base_to_strand_mapping


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
