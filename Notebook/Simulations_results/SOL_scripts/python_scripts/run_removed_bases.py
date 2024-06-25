from itertools import combinations
import numpy as np
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf, from_path
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox
from pathlib import Path
import os
from ipy_oxdna.dna_structure import DNAStructure, DNAStructureStrand, load_dna_structure, DNABase, strand_from_info
from copy import deepcopy
from ipy_oxdna.oxdna_simulation import Simulation , SimulationManager
import copy
from tqdm.auto import tqdm
# simulation of the unmodified structure for initial relaxation 

eq_steps = 1e5
prod_steps = 1e5
rel_steps = 1e5
eq_parameters = {'dt':f'0.003','steps':f'{eq_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                           'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}

prod_parameters = {'dt':f'0.003','steps':f'{prod_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                           'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}


# create relxation simulation
 
path = '/home/ava/Dropbox (ASU)/temp/Metabackbone/structure_files/six_helix_oxdna_file/modified/remove_bases'
file_dir = os.path.join(path,'1512_bp')
sim_path = '/home/ava/Dropbox (ASU)/temp/Metabackbone/metabackbone/notebooks/Simulations/simulations/removed_bases/1512_bp'
rel_dir = os.path.join(sim_path,'relaxed')

sim = Simulation(file_dir, rel_dir)
sim.build(clean_build='force')      
sim.input.swap_default_input("cpu_MC_relax")
rel_parameters = {'steps':f'{rel_steps}'}
sim.input_file(rel_parameters)
sim.oxpy_run.run(join = True)


# create equilibrium simulation

eq_dir = os.path.join(sim_path,'eq')
eq_sim = Simulation(rel_dir, eq_dir)
eq_sim.build(clean_build='force')
eq_sim.input_file(eq_parameters)
eq_sim.oxpy_run.run(join = True)



# create production simulation

prod_dir = sim_dir = os.path.join(sim_path,'prod')
prod_sim = Simulation(eq_dir, prod_dir)
prod_sim.build(clean_build='force')
prod_sim.input_file(prod_parameters)
prod_sim.oxpy_run.run(join = True)