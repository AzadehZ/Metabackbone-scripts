from __future__ import annotations
import itertools
import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Union, Generator
from .defaults import SEQ_DEP_PARAMS
from .util import rotation_matrix
import numpy as np
import sys
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf, from_path, get_traj_info, linear_read
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox
from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, TrajInfo, Configuration
import os
from ipy_oxdna.dna_structure import DNAStructure, DNAStructureStrand, load_dna_structure, DNABase, strand_from_info
from copy import deepcopy
from ipy_oxdna.oxdna_simulation import Simulation, SimulationManager
import copy
from tqdm.auto import tqdm
import random

class StructureEvolution:
    """
    The StructureEvolution class is responsible for improving DNA structures through a step-by-step process. 
    It handles the entire workflow, including making changes to the structures, running simulations, 
    and evaluating how well the structures meet the desired goals over several rounds.

    Attributes:
        current_structures (list[DNAStructure]): The list of current DNA structures being evolved.
        current_left_indices (list[list[int]]): List of left indices for each structure.
        current_right_indices (list[list[int]]): List of right indices for each structure.
        num_iterations (int): Number of iterations for the evolutionary algorithm.
        num_best_structures (int): Number of best structures to keep in each iteration.
        desired_angle (float): The target bend angle desired in the evolved DNA structures.
        tolerance (float): The acceptable tolerance for the bend angle.
        base_path (str): Base path where the modified DNA structures are saved.
        sim_base_path (str): Base path where simulation results are stored.
        sphere_radius (float): Radius of the sphere used to find and remove strands.
    
    Methods:
        run(): Executes the evolutionary algorithm over the specified number of iterations.
        _iterate_evolution(iteration): Performs one iteration of the evolutionary process.
        _evaluate_fitness(angles): Evaluates the fitness of each structure based on the bend angles.
        _select_best_mutant(angles, fitness_scores, new_structures, new_left_indices, new_right_indices): 
            Selects the best mutant structure based on fitness scores.
        _plot_iteration_results(angles, iteration): Plots the results of the current iteration.
        _update_right_left_indexes(mutants, removed_strands, left_indices, right_indices): 
            Updates the left and right indices after modifications.
    """

    def __init__(self, initial_dna_structure, left_indices, right_indices, num_iterations, num_best_structures, desired_angle, tolerance, base_path, sim_base_path, sphere_radius):
        self.current_structures = [initial_dna_structure]
        self.current_left_indices = [left_indices]
        self.current_right_indices = [right_indices]
        self.num_iterations = num_iterations
        self.num_best_structures = num_best_structures
        self.desired_angle = desired_angle
        self.tolerance = tolerance
        self.base_path = base_path
        self.sim_base_path = sim_base_path
        self.sphere_radius = sphere_radius
        self.removed_staples_dict = {}  # Dictionary to store removed staples info
        self.fitness_history = []
        self.angle_history = []

    def run(self):
        """
        Executes the evolutionary algorithm over the specified number of iterations.
        Returns the history of fitness scores and bend angles.
        """
        for iteration in range(self.num_iterations):
            self._iterate_evolution(iteration)
        return self.fitness_history, self.angle_history

    def _iterate_evolution(self, iteration):
        """
        Performs one iteration of the evolutionary process, including structure modification,
        simulation, fitness evaluation, and selection of the best mutant structures.
        
        Args:
            iteration (int): The current iteration number.
        """
        print_colored(f"\nStep 1: Iteration {iteration + 1}\n", colors['red'])
        
        new_structures, new_left_indices, new_right_indices = [], [], []
        structure_origin, removed_strands_info_all = [], []

        for i, dna in enumerate(self.current_structures):
            print_colored(f"Step 2: Processing structure {i} in iteration {iteration + 1}\n", colors['yellow'])

            # Perform necessary structure modifications
            modifications = StructureModifications()
            longest_strand, longest_strand_index = modifications.find_longest_strand(dna)
            point_pos = modifications.find_valid_point(dna, self.current_left_indices[i], self.current_right_indices[i], longest_strand)
            strands_in_sphere = modifications.find_strands_in_sphere(dna, point_pos, self.sphere_radius, exclude_strand=longest_strand_index)
            mutants, removed_strands = modifications.remove_one_strand_in_sphere(dna, point_pos, self.sphere_radius)

            removed_strands_info_all.extend(removed_strands)  # Log the strands removed within the sphere
            new_structures.extend(mutants)
            structure_origin.extend([i] * len(mutants))  # Track origin of each mutant

            print_colored(f"Step 8: {len(mutants)} mutant structures produced.\n", colors['magenta'])

            # Update left and right indices
            updated_left_indices, updated_right_indices, removed_staples_info = self._update_right_left_indexes(
                mutants, removed_strands_info_all, self.current_left_indices[i], self.current_right_indices[i]
            )

            new_left_indices.extend(updated_left_indices)
            new_right_indices.extend(updated_right_indices)

        # Export new DNA structures
        workspace = WorkSpace(self.base_path)
        export_paths = workspace.export_dna_structures(new_structures)

        # Simulate each structure
        simulations = StructureSimulations()
        angles = []
        for export_path in export_paths:
            structure_id = export_path['structure_id']
            simulations.run_simulations_for_structure(structure_id, self.base_path, self.sim_base_path)
            simulated_dna = simulations.load_simulated_structure(structure_id, self.sim_base_path)
            bend_angle = modifications.find_bend_angle(simulated_dna, self.current_left_indices, self.current_right_indices, longest_strand, point_pos)
            angles.append((structure_id, bend_angle))
        
        # Evaluate fitness and select best mutants
        fitness_scores = self._evaluate_fitness([angle for _, angle in angles])
        best_mutant = self._select_best_mutant(angles, fitness_scores, new_structures, new_left_indices, new_right_indices)

        # Update for next iteration
        self.current_structures = [best_mutant['structure']]
        self.current_left_indices = [best_mutant['left_index']]
        self.current_right_indices = [best_mutant['right_index']]
        self.fitness_history.append(fitness_scores)
        self.angle_history.append([angle for _, angle in angles])

        # Plot results
        self._plot_iteration_results(angles, iteration)

    def _evaluate_fitness(self, angles):
        """
        Evaluates the fitness of each structure based on the difference between the actual
        bend angle and the desired bend angle.

        Args:
            angles (list[float]): List of bend angles for each structure.

        Returns:
            list[float]: List of fitness scores for each structure.
        """
        return [abs(angle - self.desired_angle) for angle in angles]

    def _select_best_mutant(self, angles, fitness_scores, new_structures, new_left_indices, new_right_indices):
        """
        Selects the best mutant structure based on the fitness scores.

        Args:
            angles (list[tuple[int, float]]): List of tuples containing structure ID and bend angle.
            fitness_scores (list[float]): List of fitness scores for each structure.
            new_structures (list[DNAStructure]): List of newly generated mutant structures.
            new_left_indices (list[list[int]]): Updated left indices for each structure.
            new_right_indices (list[list[int]]): Updated right indices for each structure.

        Returns:
            dict: Dictionary containing the best mutant structure, its indices, and its angle.
        """
        sorted_mutants = sorted(zip(angles, fitness_scores, new_structures, new_left_indices, new_right_indices), key=lambda x: x[1])
        best_mutant = {
            'structure': sorted_mutants[0][2],
            'left_index': sorted_mutants[0][3],
            'right_index': sorted_mutants[0][4],
            'angle': sorted_mutants[0][0][1],
            'fitness': sorted_mutants[0][1],
        }
        return best_mutant

    def _plot_iteration_results(self, angles, iteration):
        """
        Plots the results of the current iteration, including histograms and evolution graphs.

        Args:
            angles (list[float]): List of bend angles for each structure.
            iteration (int): The current iteration number.
        """
        workspace = WorkSpace(self.base_path)
        workspace.plot_histogram([angle for _, angle in angles], self.desired_angle, iteration)
        workspace.plot_angle_evolution(self.angle_history, self.desired_angle)
        workspace.plot_best_vs_desired_angle(self.angle_history, self.desired_angle)

    def _update_right_left_indexes(self, mutants, removed_strands, left_indices, right_indices):
        """
        Updates the left and right indices for each mutant structure after modifications.

        Args:
            mutants (list[DNAStructure]): List of mutant structures.
            removed_strands (list[int]): List of indices of removed strands.
            left_indices (list[int]): List of left indices for the current structure.
            right_indices (list[int]): List of right indices for the current structure.

        Returns:
            tuple: Updated left indices, right indices, and removed strands.
        """
        updated_left_indices, updated_right_indices = [], []
        for i, mutant in enumerate(mutants):
            removed_strand_index = removed_strands[i]
            removed_bases_indices = [base.uid for base in mutant.strands[removed_strand_index]]
            max_removed_index = max(removed_bases_indices)

            new_left_indices = [idx if idx < max_removed_index else idx - len(removed_bases_indices) for idx in left_indices]
            new_right_indices = [idx if idx < max_removed_index else idx - len(removed_bases_indices) for idx in right_indices]

            updated_left_indices.append(new_left_indices)
            updated_right_indices.append(new_right_indices)
        
        return updated_left_indices, updated_right_indices, removed_strands


class StructureModifications:
    """
    Contains methods related to modifying DNA structures. This class provides functionality
    for finding the longest strand, identifying valid points, removing strands, and calculating
    bend angles within the DNA structure.

    Methods:
        find_longest_strand(dna): Finds the longest strand in the DNA structure.
        find_cross_over_in_longest_strand(longest_strand): Identifies cross-overs in the longest strand.
        find_valid_point(dna, left_indices, right_indices, longest_strand): Finds a valid point on the DNA structure for modifications.
        calculate_left_right_pos(dna, left_indices, right_indices): Calculates the center of mass positions on the left and right sides of the structure.
        is_point_far_from_crossovers(point, crossover_positions, min_distance_threshold): Checks if a point is far enough from cross-overs.
        find_strands_in_sphere(dna, point, sphere_radius): Finds strands within a defined sphere around a point.
        remove_one_strand_in_sphere(dna, point, sphere_radius): Removes one random strand from within a sphere.
        find_bases_in_sphere(dna, point, sphere_radius): Finds bases within a sphere and maps them to their strands.
        calculate_bend_angle(P, cms_left, cms_right): Calculates the bend angle at a specific point in the structure.
        find_bend_angle(dna, left_indices, right_indices, longest_strand, point_pos): Calculates the bend angle for the DNA structure.
        find_bases_around_point(dna, point, min_distance, max_distance): Finds bases around a specified point.
        calculate_center_of_mass(positions): Calculates the center of mass for a set of positions.
    """

    @staticmethod
    def find_longest_strand(dna):
        """
        Finds the longest strand in the DNA structure.

        Args:
            dna (DNAStructure): The DNA structure to analyze.

        Returns:
            tuple: The longest strand and its index.
        """
        longest_strand = None
        longest_strand_index = -1
        max_length = 0
        for index, strand in enumerate(dna.strands):
            if len(strand.bases) > max_length:
                max_length = len(strand.bases)
                longest_strand = strand
                longest_strand_index = index
        return longest_strand, longest_strand_index

    @staticmethod
    def find_cross_over_in_longest_strand(longest_strand):
        """
        Identifies cross-overs in the longest strand by evaluating base distances and index differences.

        Args:
            longest_strand (DNAStructureStrand): The longest strand in the DNA structure.

        Returns:
            tuple: The cross-over bases, the maximum index difference, and the minimum distance.
        """
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

    @staticmethod
    def find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold=2.5):
        """
        Finds a valid point on the DNA structure for modifications by ensuring it is far from cross-overs.

        Args:
            dna (DNAStructure): The DNA structure to analyze.
            left_indices (list[int]): Indices of bases on the left side.
            right_indices (list[int]): Indices of bases on the right side.
            longest_strand (DNAStructureStrand): The longest strand in the structure.
            min_distance_threshold (float): Minimum distance from cross-overs to be considered valid.

        Returns:
            np.ndarray: The valid point's position.
        """
        cms_left_side, cms_right_side, midpoint = StructureModifications.calculate_left_right_pos(dna, left_indices, right_indices)
        cross_over_bases, max_index_difference, min_distance = StructureModifications.find_cross_over_in_longest_strand(longest_strand)
        crossover_positions = [base.pos for base in cross_over_bases if base is not None]
        t = random.uniform(0, 1)
        first_P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
        if not crossover_positions:
            return first_P
        
        while True:
            t = random.uniform(0, 1)
            P = np.array(cms_left_side + t * (cms_right_side - cms_left_side))
            if StructureModifications.is_point_far_from_crossovers(P, crossover_positions, min_distance_threshold):
                return P

    @staticmethod
    def calculate_left_right_pos(dna, left_indices, right_indices):
        """
        Calculates the center of mass positions on the left and right sides of the structure.

        Args:
            dna (DNAStructure): The DNA structure to analyze.
            left_indices (list[int]): Indices of bases on the left side.
            right_indices (list[int]): Indices of bases on the right side.

        Returns:
            tuple: The center of mass positions for the left side, right side, and their midpoint.
        """
        left_pos, right_pos = [], []
        for strand in dna.strands:
            for base in strand:
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
        
        midpoint = (cms_left_side + cms_right_side) / 2
        return cms_left_side, cms_right_side, midpoint

    @staticmethod
    def is_point_far_from_crossovers(point, crossover_positions, min_distance_threshold):
        """
        Checks if a point is far enough from cross-overs to be considered valid for modifications.

        Args:
            point (np.ndarray): The position of the point to check.
            crossover_positions (list[np.ndarray]): List of crossover positions.
            min_distance_threshold (float): Minimum distance to be considered far.

        Returns:
            bool: True if the point is far from cross-overs, False otherwise.
        """
        for pos in crossover_positions:
            distance = np.linalg.norm(np.array(point) - np.array(pos))
            if distance < min_distance_threshold:
                return False
        return True

    @staticmethod
    def find_strands_in_sphere(dna, point, sphere_radius, exclude_strand=None):
        """
        Finds strands within a defined sphere around a point, optionally excluding a specific strand.

        Args:
            dna (DNAStructure): The DNA structure to analyze.
            point (np.ndarray): The center of the sphere.
            sphere_radius (float): The radius of the sphere.
            exclude_strand (int): Index of a strand to exclude from consideration.

        Returns:
            list[int]: List of indices of strands found within the sphere.
        """
        bases_in_sphere, base_to_strand_mapping = StructureModifications.find_bases_in_sphere(dna, point, sphere_radius)
        strands_in_sphere = set(base_to_strand_mapping.values())
        longest_strand, longest_strand_index = StructureModifications.find_longest_strand(dna)
        
        if exclude_strand is not None:
            strands_in_sphere_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
        return list(strands_in_sphere_to_remove)

    @staticmethod
    def remove_one_strand_in_sphere(dna, point, sphere_radius):
        """
        Removes one random strand from within a sphere centered at a given point.

        Args:
            dna (DNAStructure): The DNA structure to modify.
            point (np.ndarray): The center of the sphere.
            sphere_radius (float): The radius of the sphere.

        Returns:
            tuple: List of new DNA structures with one strand removed and list of removed strand indices.
        """
        bases_in_sphere, base_to_strand_mapping = StructureModifications.find_bases_in_sphere(dna, point, sphere_radius)
        longest_strand, longest_strand_index = StructureModifications.find_longest_strand(dna)
        strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
        strands_in_sphere = list(set(base_to_strand_mapping.values()))
        
        dna_structures = []
        removed_strands = []

        for strand_index in strands_to_remove:
            strand_list = []
            for idx, strand in enumerate(dna.strands):
                if idx != strand_index:
                    strand_list.append(strand)
                else:
                    removed_strands.append(strand_index)
            new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
            dna_structures.append(new_dna_structure)
        
        return dna_structures, removed_strands

    @staticmethod
    def find_bases_in_sphere(dna, point, sphere_radius):
        """
        Finds bases within a sphere and maps them to their corresponding strands.

        Args:
            dna (DNAStructure): The DNA structure to analyze.
            point (np.ndarray): The center of the sphere.
            sphere_radius (float): The radius of the sphere.

        Returns:
            tuple: List of base unique IDs and a mapping of bases to their strand indices.
        """
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

    @staticmethod
    def calculate_bend_angle(P, cms_left, cms_right):
        """
        Calculates the bend angle at a specific point in the structure based on the positions of
        the left and right sides.

        Args:
            P (np.ndarray): The position of the point where the bend angle is calculated.
            cms_left (np.ndarray): The center of mass of the left side.
            cms_right (np.ndarray): The center of mass of the right side.

        Returns:
            float: The calculated bend angle in degrees.
        """
        vec_left = cms_left - P
        vec_right = cms_right - P
        unit_vec_left = vec_left / np.linalg.norm(vec_left)
        unit_vec_right = vec_right / np.linalg.norm(vec_right)
        dot_product = np.dot(unit_vec_left, unit_vec_right)
        angle = np.arccos(dot_product) * (180.0 / np.pi)
        return angle

    @staticmethod
    def find_bend_angle(dna, left_indices, right_indices, longest_strand, point_pos, min_distance_threshold=2.5, min_distance=7.0, max_distance=20.0):
        """
        Calculates the bend angle for the DNA structure by finding the bases around a point
        and determining the angle formed by the left and right sides.

        Args:
            dna (DNAStructure): The DNA structure to analyze.
            left_indices (list[int]): Indices of bases on the left side.
            right_indices (list[int]): Indices of bases on the right side.
            longest_strand (DNAStructureStrand): The longest strand in the structure.
            point_pos (np.ndarray): The position of the point where the bend angle is calculated.
            min_distance_threshold (float): Minimum distance from cross-overs to be considered valid.
            min_distance (float): Minimum distance to consider for bases around the point.
            max_distance (float): Maximum distance to consider for bases around the point.

        Returns:
            float: The calculated bend angle.
        """
        left_bases, right_bases, cms_left_bases, cms_right_bases, left_base_indices, right_base_indices, left_strand_indices, right_strand_indices = StructureModifications.find_bases_around_point(dna, point_pos, min_distance, max_distance)  
        cms_left = StructureModifications.calculate_center_of_mass(left_bases)
        cms_right = StructureModifications.calculate_center_of_mass(right_bases)
        bend_angle = StructureModifications.calculate_bend_angle(point_pos, cms_left, cms_right)
        return bend_angle

    @staticmethod
    def find_bases_around_point(dna, point, min_distance, max_distance):
        """
        Finds bases around a specified point within a given distance range, categorizing them
        into left and right bases based on their position relative to the point.

        Args:
            dna (DNAStructure): The DNA structure to analyze.
            point (np.ndarray): The reference point.
            min_distance (float): Minimum distance from the point to consider.
            max_distance (float): Maximum distance from the point to consider.

        Returns:
            tuple: Lists of left and right bases, their center of mass positions, and their indices.
        """
        left_bases, right_bases = [], []
        left_base_indices, left_strand_indices = [], []
        right_base_indices, right_strand_indices = [], []
        
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
        else:
            cms_left_bases = None
        
        if right_bases:
            cms_right_bases = np.mean(right_bases, axis=0)
        else:
            cms_right_bases = None
        
        return (left_bases, right_bases, cms_left_bases, cms_right_bases, left_base_indices, right_base_indices, left_strand_indices, right_strand_indices)

    @staticmethod
    def calculate_center_of_mass(positions):
        """
        Calculates the center of mass for a set of positions.

        Args:
            positions (list[np.ndarray]): List of positions to calculate the center of mass.

        Returns:
            np.ndarray: The calculated center of mass.
        """
        if not positions:
            raise ValueError("No positions provided for center of mass calculation.")
        return np.mean(positions, axis=0)

class StructureSimulations:
    """
    Manages running and loading DNA structure simulations. This class provides methods
    to run simulations for structures, load simulated structures, and manage simulation queues.

    Methods:
        run_simulations_for_structure(structure_id, base_path, sim_base_path): Runs all simulation steps (relaxation, equilibration, production) for a given structure.
        load_simulated_structure(structure_id, sim_base_path): Loads a simulated DNA structure from the specified path.
        queue_relaxation_simulations(structures, base_path, sim_base_path): Queues and runs relaxation simulations for the given structures.
        queue_equilibration_simulations(structures, base_path, sim_base_path): Queues and runs equilibration simulations for the given structures.
        queue_production_simulations(structures, base_path, sim_base_path): Queues and runs production simulations for the given structures.
        run_all_simulations(structures, base_path, sim_base_path): Runs all stages of simulations (relaxation, equilibration, production) for a set of structures.
    """

    @staticmethod
    def run_simulations_for_structure(structure_id, base_path, sim_base_path, rel_parameters, eq_parameters, prod_parameters):
        """
        Runs all simulation steps (relaxation, equilibration, production) for a given structure.

        Args:
            structure_id (int): The ID of the structure to simulate.
            base_path (str): The base directory where structure data is stored.
            sim_base_path (str): The base directory for simulation results.
            rel_parameters (dict): Parameters for relaxation simulation.
            eq_parameters (dict): Parameters for equilibration simulation.
            prod_parameters (dict): Parameters for production simulation.
        """
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

    @staticmethod
    def load_simulated_structure(structure_id, sim_base_path):
        """
        Loads a simulated DNA structure from the specified path.

        Args:
            structure_id (int): The ID of the structure to load.
            sim_base_path (str): The base path where simulation results are stored.

        Returns:
            DNAStructure: The loaded DNA structure.
        """
        sim_path = os.path.join(sim_base_path, f'2268_bp_{structure_id}', 'prod')
        dat_path = os.path.join(sim_path, 'last_conf.dat')
        top_path = os.path.join(sim_path, '1512_bp_rmv_staples.top')

        if not os.path.isfile(dat_path):
            raise FileNotFoundError(f"Configuration file not found: {dat_path}")
        if not os.path.isfile(top_path):
            raise FileNotFoundError(f"Topology file not found: {top_path}")

        dna = load_dna_structure(top_path, dat_path)
        return dna

    @staticmethod
    def queue_relaxation_simulations(structures, base_path, sim_base_path):
        """
        Queues and runs relaxation simulations for the given structures.

        Args:
            structures (list[DNAStructure]): The list of DNA structures to simulate.
            base_path (str): The base directory where structure data is stored.
            sim_base_path (str): The base directory for simulation results.

        Returns:
            list[Simulation]: List of relaxation simulation objects.
        """
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

    @staticmethod
    def queue_equilibration_simulations(structures, base_path, sim_base_path):
        """
        Queues and runs equilibration simulations for the given structures.

        Args:
            structures (list[DNAStructure]): The list of DNA structures to simulate.
            base_path (str): The base directory where structure data is stored.
            sim_base_path (str): The base directory for simulation results.

        Returns:
            list[Simulation]: List of equilibration simulation objects.
        """
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

    @staticmethod
    def queue_production_simulations(structures, base_path, sim_base_path):
        """
        Queues and runs production simulations for the given structures.

        Args:
            structures (list[DNAStructure]): The list of DNA structures to simulate.
            base_path (str): The base directory where structure data is stored.
            sim_base_path (str): The base directory for simulation results.

        Returns:
            list[Simulation]: List of production simulation objects.
        """
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

    @staticmethod
    def run_all_simulations(structures, base_path, sim_base_path):
        """
        Runs all stages of simulations (relaxation, equilibration, production) for a set of structures.

        Args:
            structures (list[DNAStructure]): The list of DNA structures to simulate.
            base_path (str): The base directory where structure data is stored.
            sim_base_path (str): The base directory for simulation results.

        Returns:
            tuple: Lists of relaxation, equilibration, and production simulation objects.
        """
        print("Starting relaxation simulations...")
        sim_list_rel = StructureSimulations.queue_relaxation_simulations(structures, base_path, sim_base_path)
        print("Starting equilibration simulations...")
        sim_list_eq = StructureSimulations.queue_equilibration_simulations(structures, base_path, sim_base_path)
        print("Starting production simulations...")
        sim_list_prod = StructureSimulations.queue_production_simulations(structures, base_path, sim_base_path)
        return sim_list_rel, sim_list_eq, sim_list_prod
    


class WorkSpace:
    """
    Manages the workspace environment, including exporting DNA structures and plotting
    the results of the evolutionary process. This class provides utility methods for saving
    DNA structures and visualizing the distribution of bend angles.

    Attributes:
        base_path (str): The base directory where structure data is stored.

    Methods:
        export_dna_structures(new_dna_structures): Exports the modified DNA structures to the specified directory.
        plot_histogram(angles, reference_angle, iteration): Plots a histogram of bend angles for the current iteration.
        plot_angle_evolution(angle_history, desired_angle): Plots the evolution of bend angles over multiple iterations.
        plot_best_vs_desired_angle(angle_history, desired_angle): Compares the best angle found in each iteration to the desired angle.
    """

    def __init__(self, base_path):
        self.base_path = base_path

    def export_dna_structures(self, new_dna_structures):
        """
        Exports the modified DNA structures to the specified directory.

        Args:
            new_dna_structures (list[DNAStructure]): List of DNA structures to export.

        Returns:
            list[dict]: A list of dictionaries containing paths to the exported files.
        """
        output_paths = []
        for i, new_dna in enumerate(new_dna_structures):
            structure_id = i
            unique_subdir = os.path.join(self.base_path, f'structure_{structure_id}')
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

    @staticmethod
    def plot_histogram(angles, reference_angle, iteration):
        """
        Plots a histogram of bend angles for the current iteration, with the reference angle marked.

        Args:
            angles (list[float]): List of bend angles for the current iteration.
            reference_angle (float): The reference angle to be marked on the histogram.
            iteration (int): The current iteration number.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(angles, density=True, bins=10, alpha=0.6, color='blue')
        plt.axvline(reference_angle, color='red', linestyle='dashed', linewidth=2, label=f'Reference Angle: {reference_angle:.1f}')
        plt.xlabel('Angles (degrees)')
        plt.ylabel('Probability')
        plt.title(f'Iteration {iteration + 1} - Angle Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_angle_evolution(angle_history, desired_angle):
        """
        Plots the evolution of bend angles over multiple iterations, comparing each to the desired angle.

        Args:
            angle_history (list[list[float]]): A list containing bend angles from each iteration.
            desired_angle (float): The desired angle to compare against.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(angle_history) + 1), [min(angles) for angles in angle_history], marker='o', color='blue')
        plt.axhline(y=desired_angle, color='red', linestyle='dashed', label='Desired Angle')
        plt.xlabel('Iteration')
        plt.ylabel('Best Bend Angle')
        plt.title('Evolution of Bend Angle Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_best_vs_desired_angle(angle_history, desired_angle):
        """
        Compares the best angle found in each iteration to the desired angle.

        Args:
            angle_history (list[list[float]]): A list containing bend angles from each iteration.
            desired_angle (float): The desired angle to compare against.
        """
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


       
