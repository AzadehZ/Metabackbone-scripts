import os
import numpy as np
from itertools import combinations
import random
import logging
from typing import Dict, Tuple, List
from Metabackbone_functions import (load_dna_structure_files, find_longest_strand, find_cross_over_in_longest_strand, calculate_left_right_pos, find_valid_point, find_bases_around_point, calculate_center_of_mass, calculate_bend_angle, find_bend_angle, find_bases_in_sphere, remove_three_strands_in_sphere, export_dna_structures, run_all_simulations)
from ipy_oxdna.dna_structure import DNAStructure, DNAStructureStrand, load_dna_structure, DNABase, strand_from_info
from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager



def print_colored(message, color_code):
    print(f"\033[{color_code}m{message}\033[0m")
    
    
# ANSI color codes for printing colored messages
colors = {
    'blue': '34',
    'green': '32',
    'yellow': '33',
    'cyan': '36',
    'red': '31'
}

def load_simulated_structure(structure_id, sim_base_path):
    sim_path = os.path.join(sim_base_path, f'2268_bp_{structure_id}', 'prod')
    dat_path = os.path.join(sim_path, 'last_conf.dat')
    top_path = os.path.join(sim_path, '1512_bp_rmv_staples.top')

    # Check if paths exist
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"Configuration file not found: {dat_path}")
    if not os.path.isfile(top_path):
        raise FileNotFoundError(f"Topology file not found: {top_path}")

    dna = load_dna_structure(top_path, dat_path)
    return dna



def create_index_position_map(dna) -> Dict[Tuple[float, float, float], int]:
    index_position_map = {}
    for strand in dna.strands:
        for base in strand:
            index_position_map[tuple(base.pos)] = base.uid
    return index_position_map



def get_indexes_from_positions(new_positions: List[np.ndarray], index_position_map: Dict[Tuple[float, float, float], int]) -> Dict[Tuple[float, float, float], int]:
    new_index_map = {}
    for pos in new_positions:
        key = tuple(pos)
        if key in index_position_map:
            new_index_map[key] = index_position_map[key]
        else:
            # Handle the case where the position is not found
            logging.warning(f"Position {key} not found in the initial map. It may indicate that the base has moved significantly or is new.")
            raise ValueError(f"Position {key} not found in the initial map.")
    return new_index_map




def check_dna_structure(dna):
    print_colored(f"DNA type: {type(dna)}", colors['blue'])
    if hasattr(dna, 'strands'):
        print_colored("DNA has 'strands' attribute.", colors['green'])
        print_colored(f"Number of strands: {len(dna.strands)}", colors['blue'])
        for strand in dna.strands:
            print_colored(f"Strand type: {type(strand)}", colors['green'])
            print_colored(f"Number of bases in strand: {len(strand)}", colors['blue'])
            for base in strand:
                print_colored(f"Base UID: {base.uid}, Base POS: {base.pos}", colors['cyan'])
    else:
        print_colored("DNA does not have 'strands' attribute.", colors['red'])



def evaluate_fitness(angles, desired_angle, tolerance):
    fitness_scores = []
    for angle in angles:
        difference = abs(angle - desired_angle)
        fitness_score = difference
        fitness_scores.append(fitness_score)
    return fitness_scores


def run_simulations_for_structure(structure_id, base_path, sim_base_path, rel_parameters, eq_parameters, prod_parameters):
    file_dir = os.path.join(base_path, f'structure_{structure_id}')
    sim_path = os.path.join(sim_base_path, f'2268_bp_{structure_id}')
    
    # Relaxation simulation
    rel_dir = os.path.join(sim_path, 'relaxed')
    sim_relax = Simulation(file_dir, rel_dir)
    sim_relax.build(clean_build='force')
    sim_relax.input.swap_default_input("cpu_MC_relax")
    sim_relax.input_file(rel_parameters)
    
    simulation_manager = SimulationManager()
    simulation_manager.queue_sim(sim_relax)
    simulation_manager.worker_manager(gpu_mem_block=False)
    print_colored(f"Relaxation simulation for structure {structure_id} completed.", colors['green'])
    
    # Equilibration simulation
    eq_dir = os.path.join(sim_path, 'eq')
    sim_eq = Simulation(rel_dir, eq_dir)
    sim_eq.build(clean_build='force')
    sim_eq.input_file(eq_parameters)
    
    simulation_manager.queue_sim(sim_eq)
    simulation_manager.worker_manager(gpu_mem_block=False)
    print_colored(f"Equilibration simulation for structure {structure_id} completed.", colors['green'])
    
    # Production simulation
    prod_dir = os.path.join(sim_path, 'prod')
    sim_prod = Simulation(eq_dir, prod_dir)
    sim_prod.build(clean_build='force')
    sim_prod.input_file(prod_parameters)
    
    simulation_manager.queue_sim(sim_prod)
    simulation_manager.worker_manager(gpu_mem_block=False)
    print_colored(f"Production simulation for structure {structure_id} completed.", colors['green'])
    print_colored(f"All simulations for structure {structure_id} completed.\n", colors['cyan'])