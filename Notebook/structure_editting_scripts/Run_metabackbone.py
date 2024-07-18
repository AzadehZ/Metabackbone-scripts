
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Metabackbone_functions import (
    load_dna_structure_files, find_valid_point, find_bases_in_sphere,
    remove_one_strand_in_sphere, find_longest_strand, find_bend_angle,
    export_dna_structures, run_all_simulations
)
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
import random

base_path_three = "/home/ava/MetaBackbone_project/Metabackbone-scripts/structure_files/six_helix_oxdna_file/Automatically_removed_staples/1512_bp/three_staples_remvd"
base_path_two = '/home/ava/MetaBackbone_project/Metabackbone-scripts/structure_files/six_helix_oxdna_file/Automatically_removed_staples/1512_bp/two_staples_remvd'
export_path_one = '/home/ava/MetaBackbone_project/Metabackbone-scripts/structure_files/six_helix_oxdna_file/Automatically_removed_staples/1512_bp/one_staple_remvd'
first_input_path = '/home/ava/MetaBackbone_project/Metabackbone-scripts/structure_files/six_helix_oxdna_file/unmodified/1512_bp'
sim_base_path = '/home/ava/MetaBackbone_project/Metabackbone-scripts/Notebook/Simulations_results/simulations_structure/Automatically_rmvd_staples/one_staple_removed'
left_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440,1441,1442,1443,1444,1445,1446,1447,1448,1449,1450,1451,1452,1453,1454,1455,1456,1457,1458,1459,1460,1461,1462,1463,1464,1465,1466,1467,1468,1469,1470,1471,1472,1473,1474,1475,1476,1477,1478,1479,1480,1481,1482,1483,1484,1485,1486,1487,1488,1489,1490,1491,1492,1493,2161,2162,2163,2164,2165,2166,2167,2168,2169,2170,2171,2172,2173,2174,2175,2176,2177,2178,2179,2180,2185,2186,2187,2188,2190,2191,2192,2193,2194,2195,2196,2197,2198,2199,2200,2201,2202,2203,2204,2205,2206,2207,2208,2209,2210,2211,2212,2213,2214,2215,2216,2217,2218,2219,2220,2221,2222,2223,2224,2225,2226,2227,2228,2229,2230,2231,2232,2233,2234,2235,2236,2237,2238,2239,2240,2241,2242,2243,2244,2245,2246,2247,2248,2249,2250,2251,2252,2253,2254,2255,2256,2257,2258,2259,2260,2261,2262,2263,2264,2265,2266,2267,2268,2269,2270,2271,2272,2273,2274,2275,2276,2277,2278,2279,2280,2281,2282,2283,2284,2285,2286,2287,2288,2289,2290,2291,2292,2293,2294,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304,2305,2306,2307,2308,2309,2310,2311,2312,2313,2314,2315,2316,2317,2318,2319,2320,2321,2322,2323,2324,2325,2326,2327,2328,2329,2330,2331,2332,2333,2334,2335,2336,2337,2338,2339,2340,2341,2342,2343,2344,2345,2346,2347,2348,2349,2350,2351,2352,2353,2354,2355,2356,2357,2358,2359,2360,2361,2362,2363,2364,2365,2366,2367,2368,2369,2370,2371,2372,2373,2374,2375,2376,2377,2378,2379,2380,2381,2713,2714,2715,2716,2717,2748,2749,2750,2751,2752,2753,2754,2755,2756,2757,2758,2759,2760,2761,2762,2763,2764,2765,2766,2767,2768,2769,2770,2771,2772,2773,2774,2775,2776,2777,2778,2779,2780,2781,2782,2783,2784,2785,2786,2787,2788,2789,2790,2791,2792,2793,2794,2795,2796,2797,2798,2799,2800,2801,2802,2803,2804,2805,2806,2807,2808,2809,2810,2811,2812,2813,2814,2815,2816,2817,2971,2972,2973,2974,2975,2976,2977,2978,2979,2980,2981,2982,2983,2984,2985,2986,2987,2988,2989,2990,2991,2992,2993,2994,2995,2996,2997,2998,2999,3000,3001,3002,3003,3004,3005,3006,3007,3008,3009,3010,3011,3012,3013,3014,3015,3016,3017,3018,3019,3020,3021,3022,3023,3024,3025,3026,3027,3028,3029,3030,3031,3032,3033,3034,3035,3036,3037,3038,3039,3040,3041,3042,3043,3044,3045,3046,3047,3048,3049,3050,3051,3052,3053,3054,3055,3056,3057,3058,3059,3060,3061,3062,3063,3064,3065,3066,3067]
right_indices = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1555,1556,1557,1558,1559,1560,1561,1562,1563,1564,1565,1566,1567,1568,1569,1570,1571,1572,1573,1574,1575,1576,1577,1578,1579,1580,1581,1582,1583,1584,1585,1586,1587,1588,1589,1590,1591,1592,1593,1594,1595,1596,1597,1598,1599,1600,1601,1602,1603,1604,1605,1606,1607,1608,1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1625,1626,1627,1628,1629,1630,1631,1632,1633,1634,1635,1636,1637,1638,1639,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1652,1653,1654,1655,1656,1657,1658,1659,1660,1661,1662,1663,1664,1665,1666,1667,1668,1669,1670,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715,1716,1717,1718,1719,1720,1721,1722,1723,1724,1725,1726,1727,1728,1729,1730,1736,1737,1738,1739,1740,2382,2383,2384,2385,2386,2387,2388,2389,2390,2391,2392,2393,2394,2395,2396,2397,2398,2399,2400,2401,2402,2403,2404,2405,2406,2407,2408,2409,2410,2411,2412,2413,2414,2415,2416,2417,2418,2419,2420,2421,2422,2423,2424,2425,2426,2427,2428,2429,2430,2431,2432,2433,2434,2435,2436,2437,2438,2439,2440,2441,2442,2443,2444,2445,2446,2447,2448,2449,2450,2451,2452,2453,2454,2455,2456,2457,2458,2459,2460,2461,2462,2463,2464,2465,2466,2467,2468,2469,2470,2471,2472,2473,2474,2475,2476,2477,2478,2479,2480,2481,2482,2483,2484,2485,2486,2487,2488,2489,2490,2491,2492,2493,2494,2495,2496,2497,2498,2499,2500,2501,2502,2513,2514,2515,2516,2517,2518,2519,2520,2521,2522,2525,2526,2527,2528,2529,2530,2531,2532,2533,2534,2535,2536,2537,2818,2819,2820,2821,2822,2823,2824,2825,2826,2827,2828,2829,2830,2831,2832,2833,2834,2835,2836,2837,2838,2839,2840,2841,2842,2843,2844,2845,2846]

sphere_radius = 3.5
eq_steps = 1e5
prod_steps = 1e5
rel_steps = 1e3
min_distance_threshold = 2.5
min_distance = 7.0
max_distance = 20.0
eq_parameters = {'dt':f'0.003','steps':f'{eq_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                 'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}

prod_parameters = {'dt':f'0.003','steps':f'{prod_steps}','print_energy_every': f'1e5', 'interaction_type': 'DNA2',
                   'print_conf_interval':f'1e5', 'fix_diffusion':'false', 'T':f'20C','max_density_multiplier':f'50'}
rel_parameters = {'steps': f'{rel_steps}', 'max_backbone_force': '200', 'max_backbone_force_far': '200'}


print("Loading DNA structure...")
dna = load_dna_structure_files(first_input_path)
print("DNA structure loaded.")

print("Finding longest strand...")
longest_strand, longest_strand_index = find_longest_strand(dna)
print(f"Longest strand found: Index {longest_strand_index}, Length {len(longest_strand.bases)}")

print("Finding a valid point for bend angle calculation...")
point = find_valid_point(dna, left_indices, right_indices, longest_strand, min_distance_threshold)
print(f"Valid point found at coordinates: {point}")

print(f"Finding bases and strands within a sphere of radius {sphere_radius} centered at {point}...")
bases_in_sphere, base_to_strand_mapping = find_bases_in_sphere(dna, point, sphere_radius)
strands_to_remove = set(base_to_strand_mapping.values()) - {longest_strand_index}
strands_in_sphere = list(set(base_to_strand_mapping.values()))

print("Strand indices in the sphere:", strands_in_sphere)
print("Strand indices to be removed:", list(strands_to_remove))

best_angle_diff = float('inf')
best_structure = None
best_bend_angle = None
best_point_pos = None

for strand_index in strands_to_remove:
    # Create new structure by removing one strand
    print(f"Removing strand {strand_index} and creating new DNA structure...")
    strand_list = [strand for idx, strand in enumerate(dna.strands) if idx != strand_index]
    new_dna_structure = DNAStructure(strand_list, dna.time, dna.box, dna.energy)
    
    # Export new DNA structure
    file_dir = os.path.join(export_path_one, f'structure_{strand_index}')
    sim_dir = os.path.join(sim_base_path, f'structure_{strand_index}')
    print('sim_dir:', sim_dir)
    print('unique_subdir:', file_dir)
    os.makedirs(file_dir, exist_ok=True)
    dat_path = os.path.join(file_dir, '1512_bp_rmv_staples.dat')
    top_path = os.path.join(file_dir, '1512_bp_rmv_staples.top')
    new_dna_structure.export_top_conf(Path(top_path), Path(dat_path))
    print(f"New DNA structure exported to {file_dir}")
    
    rel_dir = os.path.join(sim_dir, 'relaxed')
    eq_dir = os.path.join(sim_dir, 'eq')
    prod_dir = os.path.join(sim_dir, 'prod')
    
    print(f"Running simulations for structure with strand {strand_index} removed...")
    # Create relaxation simulation 
    print("Starting relaxation simulation setup...")
    sim = Simulation(file_dir, rel_dir)
    sim.build(clean_build='force')  
    sim.input.swap_default_input("cpu_MC_relax")
    rel_parameters = {'steps': f'{rel_steps}'}
    sim.input_file(rel_parameters)
    # sim.add_force_file()
    sim.oxpy_run.run(join=True)
    print("Relaxation simulation completed.")
    
    # Create equilibrium simulation
    print("Starting equilibrium simulation setup...")
    eq_sim = Simulation(rel_dir, eq_dir)
    eq_sim.build(clean_build='force')
    eq_sim.input_file(eq_parameters)
    # eq_sim.add_force_file()
    eq_sim.oxpy_run.run(join=True)
    print("Equilibrium simulation completed.")
    
    # Create production simulation 
    print("Starting production simulation setup...")
    prod_sim = Simulation(eq_dir, prod_dir)
    prod_sim.build(clean_build='force')
    prod_sim.input_file(prod_parameters)
    # prod_sim.add_force_file()
    prod_sim.oxpy_run.run(join=True)
    print("Production simulation completed.") 

    print(f"Simulations completed for structure with strand {strand_index} removed.")
    
    # Load simulated structure
    dat_path = os.path.join(prod_dir, 'trajectory.dat')
    top_path = os.path.join(prod_dir, '1512_bp_rmv_staples.top')
    simulated_dna = load_dna_structure(top_path, dat_path)
    
    
    # Measure bend angle for simulated structure
    point_pos, bend_angle = find_bend_angle(simulated_dna, left_indices, right_indices, longest_strand, min_distance_threshold, min_distance, max_distance)
    print(f"Bend angle measured for structure with strand {strand_index} removed: {bend_angle:.2f} degrees")
    
#     # Compare angle with reference angle
#     ref_angle = 0  # Assuming reference angle is zero for comparison
#     angle_diff = abs(bend_angle - ref_angle)
#     if angle_diff < best_angle_diff:
#         best_angle_diff = angle_diff
#         best_structure = new_dna_structure
#         best_bend_angle = bend_angle
#         best_point_pos = point_pos
#         print(f"New best structure found with angle difference: {best_angle_diff:.2f} degrees")

# print(f"Best structure found with angle difference: {best_angle_diff:.2f} degrees")

# def visualize_dna_structure(dna_structure, point_pos, bend_angle):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     for strand in dna_structure.strands:
#         x = [base.pos[0] for base in strand.bases]
#         y = [base.pos[1] for base in strand.bases]
#         z = [base.pos[2] for base in strand.bases]
#         ax.plot(x, y, z, label='Strand')
    
#     # Plot the point where bend angle is calculated
#     ax.scatter(point_pos[0], point_pos[1], point_pos[2], color='r', s=100, label='Bend Point')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f'Bend Angle: {bend_angle:.2f} degrees')
#     plt.legend()
#     plt.show()

# # Visualize the best structure
# visualize_dna_structure(best_structure, best_point_pos, best_bend_angle)