# dna_analysis.py

import os
import random
import numpy as np
from pathlib import Path
from itertools import combinations
import sys
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf, from_path
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox
from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, TrajInfo
from pathlib import Path
from ipy_oxdna.dna_structure import DNAStructure, DNAStructureStrand, load_dna_structure, DNABase, strand_from_info
from copy import deepcopy
from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager
import copy
from tqdm.auto import tqdm
from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager


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
    for strand in dna.strands:
        for base in strand:
            if base.uid in right_indices:
                right_pos.append(base.pos)
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

def find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold=3):
    cms_left_side, cms_right_side = calculate_left_right_pos(dna, left_indices, right_indices)
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
    for strand in dna.strands:
        for base in strand:
            distance = np.linalg.norm(np.array(base.pos) - np.array(point))
            if min_distance < distance < max_distance:
                if base.pos[0] < point[0]:
                    left_bases.append(base.pos)
                else:
                    right_bases.append(base.pos)
    return left_bases, right_bases

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

def find_bend_angle(dna, left_indices, right_indices, longest_strand, min_distance_threshold=2.0, min_distance=5.0, max_distance=10.0):
    point_pos = find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold)
    left_bases, right_bases = find_bases_around_point(dna, point_pos, min_distance, max_distance)
    cms_left = calculate_center_of_mass(left_bases)
    cms_right = calculate_center_of_mass(right_bases)
    bend_angle = calculate_bend_angle(point_pos, cms_left, cms_right)
    return point_pos, bend_angle

def calculate_angles_for_all_structures(dna_list, left_indices, right_indices, min_distance_threshold=2.0, min_distance=5.0, max_distance=10.0):
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

def queue_relaxation_simulations(structures, base_path, sim_base_path):
    rel_steps = 1e3
    rel_parameters = {'steps': f'{rel_steps}', 'max_backbone_force': '200', 'max_backbone_force_far': '200'}
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
    eq_steps = 1e5
    eq_parameters = {'dt':f'0.003','steps':f'{eq_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                 'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}
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
    prod_steps = 1e5
    prod_parameters = {'dt':f'0.003','steps':f'{prod_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                   'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}
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

def main(base_path, sim_base_path, left_indices, right_indices, target_angle=90, tolerance=5):
    initial_dna = load_dna_structure_files(base_path)
    modified_structures = [initial_dna]
    results = []
    sphere_radius = 3.5  # Define your sphere radius
    while True:
        new_structures = []
        for dna in modified_structures:
            P = find_valid_point(dna, left_indices, right_indices, find_longest_strand(dna)[0])
            new_dna_structures = remove_one_strand_in_sphere(dna, P, sphere_radius)
            new_structures.extend(new_dna_structures)
        if not new_structures:
            break
        output_paths = export_dna_structures(new_structures, base_path)
        sim_list_rel, sim_list_eq, sim_list_prod = run_all_simulations(new_structures, base_path, sim_base_path)
        for dna in new_structures:
            point_pos, bend_angle = find_bend_angle(dna, left_indices, right_indices, find_longest_strand(dna)[0])
            results.append((point_pos, bend_angle))
            if abs(bend_angle - target_angle) <= tolerance:
                print(f"Achieved target bend angle: {bend_angle} at point {point_pos}")
                return results
        modified_structures = new_structures
    return results
