import os
import numpy as np
from itertools import combinations
import random
import logging
from typing import Dict, Tuple, List
from Metabackbone_functions import (load_dna_structure_files, find_longest_strand, find_cross_over_in_longest_strand, calculate_left_right_pos, find_valid_point, find_bases_around_point, calculate_center_of_mass, calculate_bend_angle, find_bend_angle, find_bases_in_sphere, remove_three_strands_in_sphere, export_dna_structures, run_all_simulations)
from ipy_oxdna.dna_structure import DNAStructure, DNAStructureStrand, load_dna_structure, DNABase, strand_from_info
from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager
import seaborn as sns
import matplotlib.pyplot as plt



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


def find_strands_in_sphere(dna, point, sphere_radius, exclude_strand = None):
    # Find bases within the sphere and map them to their strands
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    
    # Identify the strands that have bases within the sphere
    strands_in_sphere = set(base_to_strand_mapping.values())
    longest_strand, longest_strand_index = find_longest_strand(dna)
    
    # Optionally exclude a specific strand by comparing their unique identifiers or indices
    if exclude_strand is not None:
        strands_in_sphere_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}

    return list(strands_in_sphere_to_remove)


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
    print_colored(f"All simulations for mutant {structure_id} completed.\n", colors['cyan'])
    
    
def stored_removed_strands(dna, removed_strands_info):
    summaries = []
    for i, strands in enumerate(removed_strands_info):
        removed_bases = []
        for strand in strands:
            removed_bases.extend([base.uid for base in dna.strands[strand]])
        summary = f"For structure_{i}, staples with the indexes {', '.join(map(str, strands))} were removed."
        summaries.append((summary, removed_bases))
    return summaries

def update_right_left_indexes(mutants, removed_strands, left_indices, right_indices):
    updated_left_indices = []
    updated_right_indices = []
    for i, mutant in enumerate(mutants):
        removed_strand_index = removed_strands[i]
        removed_bases_indices = [base.uid for base in mutant.strands[removed_strand_index]]
        max_removed_index = max(removed_bases_indices)
        
        new_left_indices = []
        for idx in left_indices:
            if idx < max_removed_index:
                new_left_indices.append(idx)
            else:
                new_left_indices.append(idx - len(removed_bases_indices))
        print(f"Updated left indices for mutant {i+1}: {new_left_indices}")        
        
        new_right_indices = []
        for idx in right_indices:
            if idx < max_removed_index:
                new_right_indices.append(idx)
            else:
                new_right_indices.append(idx - len(removed_bases_indices))        
        print(f"Updated right indices for mutant {i+1}: {new_right_indices}")
        
        updated_left_indices.append(new_left_indices)
        updated_right_indices.append(new_right_indices)
        
    return updated_left_indices, updated_right_indices, removed_strands 

def get_removed_strands_by_index(dna: DNAStructure, removed_strands_info: list[int]) -> list[DNAStructureStrand]:
    """
    Parameters:
        dna (DNAStructure): The DNA structure object from which strands were removed.
        removed_strands_info (list[int]): A list of strand indices that were removed.
    
    Returns:
        list[DNAStructureStrand]: A list of DNAStructureStrand objects corresponding to the removed strands.
    """
    removed_strands = []
    
    for strand_index in removed_strands_info:
        if 0 <= strand_index < dna.get_num_strands():
            removed_strands.append(dna.get_strand(strand_index))
        else:
            print(f"Warning: Invalid strand index {strand_index}. Skipping.")
    
    return removed_strands


def evolutionary_algorithm(initial_dna_structure, left_indices, right_indices, num_iterations, num_best_structures, desired_angle, tolerance, base_path, sim_base_path, sphere_radius):
    current_structures = [initial_dna_structure]
    current_left_indices = [left_indices]
    current_right_indices = [right_indices]
    removed_staples_dict = {}  # Dictionary to store removed staples info
    
    # Initialize lists to store the results
    left_indices_list = []
    right_indices_list = []
    
    fitness_history = []
    angle_history = []
    removed_staples_info_all_iterations = []

    for iteration in range(num_iterations):
        print_colored(f"\nStep 1: Iteration {iteration + 1}\n", colors['red'])
        
        new_structures = []
        new_left_indices = []
        new_right_indices = []
        removed_strands_info_all = []
        structure_origin = []  # To keep track of which structure each mutant came from
        
        for i, dna in enumerate(current_structures):
            print_colored(f"Step 2: Processing structure {i} in iteration {iteration + 1}\n", colors['yellow'])

            # Step 2: Find the longest strand in the DNA structure
            longest_strand, longest_strand_index = find_longest_strand(dna)
            print_colored(f'Step 3: Longest strand: {longest_strand}', colors['cyan'])
            print_colored(f'Longest strand index: {longest_strand_index}\n', colors['cyan'])

            # Step 3: Find a valid point on the structure
            point_pos = find_valid_point(dna, current_left_indices[i], current_right_indices[i], longest_strand)
            print_colored(f'Step 4: Found a valid point in the DNA structure: {point_pos}\n', colors['blue'])

            # Step 4: Define a sphere centered at the valid point
            # The sphere is conceptually defined here as the area around `point_pos` within `sphere_radius`
            print_colored(f'Step 5: Defined a sphere around the valid point with radius {sphere_radius}\n', colors['magenta'])
        
            # Step 5: Find the strands within that sphere, excluding the longest strand
            strands_in_sphere = find_strands_in_sphere(dna, point_pos, sphere_radius, exclude_strand = longest_strand_index)
            print_colored(f'Step 6: Number of strands found in the sphere: {len(strands_in_sphere)}\n', colors['green'])
        
            # Step 6: Remove one random staple from the staples within the sphere
            mutants, removed_strands = remove_one_strand_in_sphere(dna, point_pos, sphere_radius)
            removed_strands_info_all.extend(removed_strands)  # Log the strands removed within the sphere
            print_colored(f'Step 7: Removed strands within the sphere. Number of mutants generated: {len(mutants)}\n', colors['red'])

            # Step 7: Generate mutant structures
            new_structures.extend(mutants)
            structure_origin.extend([i] * len(mutants))  # Keep track of the origin
            print_colored(f"Step 8: By modifying structure {i}, {len(mutants)} mutant structures were produced.\n", colors['magenta'])

            # Step 8: Update the left and right indices for each mutant structure
            print_colored("Step 9: Updating left and right indices for each mutant...", colors['green'])
            updated_left_indices, updated_right_indices, removed_staples_info = update_right_left_indexes(mutants, removed_strands_info_all, current_left_indices[i], current_right_indices[i])

            # Store the updated indices
            new_left_indices.extend(updated_left_indices)
            new_right_indices.extend(updated_right_indices)
            print_colored(f'Step 10: Updated left indices: {updated_left_indices}', colors['cyan'])
            print_colored(f'Updated right indices: {updated_right_indices}\n', colors['cyan'])

        # Step 9: Export new DNA structures
        print_colored(f"Step 11: Exporting DNA structures...", colors['yellow'])
        export_paths = export_dna_structures(new_structures, base_path)
        print_colored(f"Exported DNA structures: {export_paths}\n", colors['yellow'])
    
        # Step 10: Simulate each modified structure
        for export_path in export_paths:
            structure_id = export_path['structure_id']
            print_colored(f"Step 12: Starting simulations for structure {structure_id}...", colors['red'])
            run_simulations_for_structure(structure_id, base_path, sim_base_path, rel_parameters, eq_parameters, prod_parameters)
            print_colored(f"Simulation completed for structure {structure_id}\n", colors['red'])
            
        # Step 11: Measure the angle at the joint for each mutant after simulation
        angles = []
        for export_path in export_paths:
            structure_id = export_path['structure_id']
            simulated_dna = load_simulated_structure(structure_id, sim_base_path)
            bend_angle = find_bend_angle(simulated_dna, left_indices, right_indices, longest_strand, point_pos)
            angles.append((structure_id, bend_angle))
            print_colored(f'Step 13: Structure {structure_id} - Bend Angle: {bend_angle}\n', colors['blue'])
        
        # Step 12: Evaluate fitness
        print_colored(f"Step 14: Evaluating fitness for each structure...", colors['green'])
        fitness_scores = evaluate_fitness([angle for _, angle in angles], desired_angle, tolerance)
        print_colored(f'Fitness scores: {fitness_scores}\n', colors['green'])
        
        # Step 13: Select the best mutants based on fitness scores
        print_colored(f"Step 15: Selecting the best mutants based on fitness scores...", colors['magenta'])
        sorted_mutants = sorted(zip(angles, fitness_scores, new_structures, new_left_indices, new_right_indices), key=lambda x: x[1])
        best_mutant = sorted_mutants[0]
        best_structure_id_angle, best_fitness_score, best_unsimulated_structure, best_left_index, best_right_index = best_mutant
        best_angle = best_structure_id_angle[1]
        best_structure_id = best_structure_id_angle[0]
        print_colored(f'Best mutant structure ID: {best_structure_id} with angle: {best_angle} and fitness score: {best_fitness_score}\n', colors['magenta'])
        
        # Check if the best mutant achieves the desired angle within tolerance
        if abs(best_angle - desired_angle) <= tolerance:
            print_colored(f"Step 16: Best mutant structure ID {best_structure_id} has achieved the desired angle within tolerance.\n", colors['green'])
        
        # Update the current structures and indices for the next iteration
        current_structures = [best_unsimulated_structure]
        current_left_indices = [best_left_index]
        current_right_indices = [best_right_index]
        
        # Store the fitness and angle history
        fitness_history.append(fitness_scores)
        angle_history.append([angle for _, angle in angles])
        
        # Step 14: Plot histogram for this iteration
        print_colored(f"Step 17: Plotting histograms and evolution graphs...", colors['blue'])
        plot_histogram([angle for _, angle in angles], desired_angle, iteration)
        plot_angle_evolution(angle_history, desired_angle)
        plot_best_vs_desired_angle(angle_history, desired_angle)
        print_colored(f"Iteration {iteration + 1} completed.\n", colors['yellow'])
        
    return fitness_history, angle_history



    
  
    
def find_symmetric_strands(dna, point, sphere_radius, num_strands=3):
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    strand_distances = {}

    # Calculate the distance of each strand from the point
    for strand_index in set(base_to_strand_mapping.values()):
        strand_bases = [base for base in dna.strands[strand_index]]
        avg_distance = np.mean([np.linalg.norm(np.array(base.pos) - point) for base in strand_bases])
        strand_distances[strand_index] = avg_distance

    # Sort strands by distance and select symmetrically
    sorted_strands = sorted(strand_distances.items(), key=lambda x: x[1])
    
    # Select strands symmetrically
    selected_strands = []
    i, j = 0, len(sorted_strands) - 1
    while len(selected_strands) < num_strands and i <= j:
        if len(selected_strands) < num_strands:
            selected_strands.append(sorted_strands[i][0])
            i += 1
        if len(selected_strands) < num_strands:
            selected_strands.append(sorted_strands[j][0])
            j -= 1

    return selected_strands


def remove_symmetric_strands_in_sphere(dna, point, sphere_radius, num_strands=3):
    symmetric_collections = find_multiple_symmetric_strand_collections(dna, point, sphere_radius, num_strands)
    new_structures = []
    removed_strands_info = []

    for collection in symmetric_collections:
        strand_list = [strand for idx, strand in enumerate(dna.strands) if idx not in collection]
        new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
        new_structures.append(new_dna_structure)
        removed_strands_info.append(collection)

    return new_structures, removed_strands_info


def find_multiple_symmetric_strand_collections(dna, point, sphere_radius, num_strands=3):
    bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
    strand_distances = {}

    # Calculate the distance of each strand from the point
    for strand_index in set(base_to_strand_mapping.values()):
        strand_bases = [base for base in dna.strands[strand_index]]
        avg_distance = np.mean([np.linalg.norm(np.array(base.pos) - point) for base in strand_bases])
        strand_distances[strand_index] = avg_distance

    # Sort strands by distance
    sorted_strands = sorted(strand_distances.items(), key=lambda x: x[1])
    
    # Find multiple sets of symmetric strands
    symmetric_collections = []
    n = len(sorted_strands)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if len(symmetric_collections) < num_strands:
                    symmetric_collections.append([sorted_strands[i][0], sorted_strands[j][0], sorted_strands[k][0]])
                else:
                    break

    return symmetric_collections


def plot_histogram(angles, reference_angle, iteration):
    
    plt.figure(figsize=(8, 6))

    # Create a histogram of the angles with density=True to normalize it
    plt.hist(angles, density=True, bins=10, alpha=0.6, color='blue')

    # Add the reference angle as a vertical line
    plt.axvline(reference_angle, color='red', linestyle='dashed', linewidth=2, label=f'Reference Angle: {reference_angle:.1f}')

    # Add labels and title
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Probability')
    plt.title(f'Iteration {iteration + 1} - Angle Distribution')
    plt.legend()

    # Show grid for better readability
    plt.grid(True)
    
    # Show the plot
    plt.show()


def plot_angle_evolution(angle_history, desired_angle):
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(angle_history) + 1), [min(angles) for angles in angle_history], marker='o', color='blue')
    plt.axhline(y=desired_angle, color='red', linestyle='dashed', label='Desired Angle')
    plt.xlabel('Iteration')
    plt.ylabel('Best Bend Angle')
    plt.title('Evolution of Bend Angle Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_best_vs_desired_angle(angle_history, desired_angle):
    import matplotlib.pyplot as plt

    best_angles = [min(angles) for angles in angle_history]

    plt.figure(figsize=(8, 6))
    plt.scatter(range(1, len(angle_history) + 1), best_angles, color='blue', label='Best Angle')
    plt.axhline(y=desired_angle, color='red', linestyle='dashed', label='Desired Angle')
    plt.xlabel('Iteration')
    plt.ylabel('Angle (degrees)')
    plt.title('Best Angle vs. Desired Angle Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
