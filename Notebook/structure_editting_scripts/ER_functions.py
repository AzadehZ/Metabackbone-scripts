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
            print_colored(f"Position {key} not found in the initial map.", colors['red'])
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
    return [abs(angle - desired_angle) for angle in angles]

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
    
    
# Function to create new left and right indices after removing staples
# def update_indices_for_selected_structures(selected_structures, original_structure, left_indices, right_indices, removed_staples_info):
#     new_indices = []

#     for structure, removed_strands in zip(selected_structures, removed_staples_info):
#         new_left_indices = []
#         new_right_indices = []

#         if isinstance(removed_strands, list) and all(isinstance(rs, list) for rs in removed_strands):
#             removed_bases_indices = [base.uid for strand in removed_strands for base in strand]
#         else:
#             removed_bases_indices = removed_strands

#         max_removed_index = max(removed_bases_indices) if removed_bases_indices else -1

#         for index in left_indices:
#             if index < max_removed_index:
#                 new_left_indices.append(index)
#             else:
#                 new_left_indices.append(index - len(removed_bases_indices))

#         for index in right_indices:
#             if index < max_removed_index:
#                 new_right_indices.append(index)
#             else:
#                 new_right_indices.append(index - len(removed_bases_indices))

#         new_indices.append((new_left_indices, new_right_indices))

#     return new_indices

# Function to create new left and right indices after removing staples
def update_indices_for_selected_structures(selected_structures, original_structure, left_indices, right_indices, removed_staples_info):
    new_indices = []

    for structure, removed_strands in zip(selected_structures, removed_staples_info):
        new_left_indices = []
        new_right_indices = []

        removed_bases_indices = [base.uid for strand in removed_strands for base in original_structure.strands[strand]]
        max_removed_index = max(removed_bases_indices) if removed_bases_indices else -1

        for index in left_indices:
            if index < max_removed_index:
                new_left_indices.append(index)
            else:
                new_left_indices.append(index - len(removed_bases_indices))

        for index in right_indices:
            if index < max_removed_index:
                new_right_indices.append(index)
            else:
                new_right_indices.append(index - len(removed_bases_indices))

        new_indices.append((new_left_indices, new_right_indices))

    return new_indices


def evolutionary_algorithm(initial_dna_structure, left_indices, right_indices, num_iterations, num_best_structures, desired_angle, tolerance, base_path, sim_base_path, sphere_radius):
    current_structures = [initial_dna_structure]
    removed_staples_dict = {}  # Dictionary to store removed staples info
    
    for iteration in range(num_iterations):
        print_colored(f"Iteration {iteration + 1}", colors['yellow'])
        
        new_structures = []
        removed_strands_info_all = []
        
        for dna in current_structures:
            # Step 1: Find a valid point in the DNA structure
            print_colored("Step 1: Finding a valid point in the DNA structure...", colors['red'])
            longest_strand, _ = find_longest_strand(dna)
            point_pos = find_valid_point(dna, left_indices, right_indices, longest_strand)
            print_colored(f'Found a valid point in the DNA structure: {point_pos}', colors['green'])
            
            # Step 2: Remove three random staples within a sphere around the point
            print_colored("Step 2: Removing three random staples within a sphere around the point...", colors['blue'])
            mutants, removed_strands_info = remove_three_strands_in_sphere(dna, point_pos, sphere_radius)
            removed_strands_info_all.extend(removed_strands_info)
            new_structures.extend(mutants)
            print_colored(f"Removed three random staples. Number of new structures: {len(mutants)}", colors['cyan'])
        
        # Determine the number of mutants based on the number of new structures created
        num_mutants = len(new_structures)
        print_colored(f"Iteration {iteration + 1}: Generated {num_mutants} new structures.", colors['blue'])
        
        # Step 3: Export new DNA structures
        print_colored("Step 3: Exporting DNA structures...", colors['magenta'])
        export_paths = export_dna_structures(new_structures, base_path)
        # print_colored("Exported DNA structures.", colors['yellow'])
        print_colored(f"Export paths: {export_paths}", colors['red'])
        
        # Step 4: Simulate each modified structure
        print_colored("Step 4: Simulating each modified structure...", colors['green'])
        for export_path in export_paths:
            structure_id = export_path['structure_id']
            print_colored(f"Starting simulations for structure {structure_id}...", colors['red'])
            run_simulations_for_structure(structure_id, base_path, sim_base_path, rel_parameters, eq_parameters, prod_parameters)
            # print_colored(f"Simulations for structure {structure_id} completed.", colors['blue'])
        
        # Step 5: Measure the angle at the joint for each mutant after simulation
        print_colored("Step 5: Measuring the angle at the joint for each mutant after simulation...", colors['cyan'])
        angles = []
        for export_path in export_paths:
            structure_id = export_path['structure_id']
            simulated_dna = load_simulated_structure(structure_id, sim_base_path)
            bend_angle = find_bend_angle(simulated_dna, left_indices, right_indices, longest_strand, point_pos)
            angles.append((structure_id, bend_angle))
            print_colored(f"Measured bend angle for structure {structure_id}: {bend_angle} degrees.", colors['magenta'])
        
        # Step 6: Evaluate fitness
        print_colored("Step 6: Evaluating fitness of mutants...", colors['yellow'])
        fitness_scores = evaluate_fitness([angle for _, angle in angles], desired_angle, tolerance)
        # print_colored("Evaluated fitness of mutants.", colors['red'])
        print_colored(f"Fitness scores: {fitness_scores}", colors['green'])
        
        # Step 7: Select the best mutants based on fitness scores
        print_colored("Step 7: Selecting the best mutants based on fitness scores...", colors['blue'])
        sorted_mutants = sorted(zip(angles, fitness_scores), key=lambda x: x[1])
        
        # Display results for all structures
        print_colored("Results for all structures:", colors['cyan'])
        for (structure_id, angle), fitness_score in sorted_mutants:
            print_colored(f"Structure ID: {structure_id}, Bend Angle: {angle}, Fitness Score: {fitness_score}", colors['magenta'])

        # Select the best mutants
        best_mutants = [new_structures[i] for i, (_, _) in enumerate(sorted_mutants[:num_best_structures])]
        best_angles = [angle for (_, angle), _ in sorted_mutants[:num_best_structures]]
        best_removed_strands_info = [removed_strands_info_all[i] for i, (_, _) in enumerate(sorted_mutants[:num_best_structures])]
        
        # Display results for the selected best structures
        print_colored(f"Selected the best {num_best_structures} mutants:", colors['yellow'])
        for i, (angle, fitness_score) in enumerate(zip(best_angles, [fs for _, fs in sorted_mutants[:num_best_structures]])):
            print_colored(f"Selected Structure {i}: Bend Angle: {angle}, Fitness Score: {fitness_score}", colors['red'])

        # Step 8: Store the removed staples info for the best mutants
        print_colored("Step 8: Storing the removed staples info for the best mutants...", colors['green'])
        for i, (structure_id, _) in enumerate(sorted_mutants[:num_best_structures]):
            removed_staples_dict[structure_id] = best_removed_strands_info[i]
            summary, removed_bases = stored_removed_strands(initial_dna_structure, [best_removed_strands_info[i]])[0]
            print_colored(f"Structure ID: {structure_id} - {summary}", colors['blue'])
            print_colored(f"Removed bases: {removed_bases}", colors['cyan'])
        
        # Check if the best angle is within the desired tolerance
        if any(abs(angle - desired_angle) <= tolerance for angle in best_angles):
            print_colored("Desired angle achieved within tolerance. Stopping evolution process.", colors['magenta'])
            break
        
        # Step 9: Update the current structures with the best mutants for the next iteration
        print_colored("Step 9: Updating the current structures with the best mutants for the next iteration...", colors['yellow'])
        current_structures = best_mutants
        print_colored(f"Updated current structures for iteration {iteration + 1}.", colors['red'])
        
        # Step 10: Update left and right indices for each selected structure
        print_colored("Step 10: Updating left and right indices for each selected structure...", colors['green'])
        new_indices_list = update_indices_for_selected_structures(current_structures, initial_dna_structure, left_indices, right_indices, best_removed_strands_info)
        left_indices, right_indices = new_indices_list[0]
    
    print_colored("Evolutionary algorithm completed.", colors['red'])