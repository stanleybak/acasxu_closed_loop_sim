'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time

This version uses multiprocessing pool for more simulations
'''

import time
import multiprocessing
import argparse

import numpy as np

from acasxu_dubins import State, make_random_input, plot, state7_to_state5, run_network

def sim_single(seed, intruder_can_turn):
    """run single simulation and return min_dist"""

    rv = np.inf

    if seed % 1000 == 0:
        print(f"{(seed//1000) % 10}", end='', flush=True)
    elif seed % 100 == 0:
        print(".", end='', flush=True)

    init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)

    v_own = init_velo[0]
    v_int = init_velo[1]

    # reject start states where initial command is not clear-of-conflict
    state5 = state7_to_state5(init_vec, v_own, v_int)

    if state5[0] > 60760:
        command = 0 # rho exceeds network limit
    else:
        res = run_network(State.nets[0], state5)
        command = np.argmin(res)

    if command == 0:
        # run the simulation
        s = State(init_vec, init_velo[0], init_velo[1], save_states=False)
        s.simulate(cmd_list)

        rv = s.min_dist

    return rv

def main():
    'main entry point'

    # dt = 0.05
    #Did 1000000 parallel sims in 794.3 secs (0.794ms per sim)
    #Rejected 58480 sims. 5.848%
    #Collision in 3 sims. 0.0003%
    #Seed 350121 has min_dist 325.4ft

    # parse arguments
    parser = argparse.ArgumentParser(description='Run ACASXU Dublins model simulator.')
    parser.add_argument("--save-mp4", action='store_true', default=False, help="Save plotted mp4 files to disk.")
    args = parser.parse_args()

    save_mp4 = args.save_mp4
    intruder_can_turn = False

    # home laptop (dt=0.05): 10000000 parallel sims take 5714.9 secs (0.571ms per sim)
    num_sims = 1000000 * 100 # 100 million, estimated runtime 6 hours
    batch_size = 50000

    remaining_sims = num_sims
    completed_sims = 0

    collision_dist = 500
    min_dist = np.inf
    min_seed = -1
    num_collisions = 0
    collision_seeds = []
    num_rejected = 0
    start = time.perf_counter()

    print(f"Running {num_sims} parallel simulations in batches of {batch_size}...")

    with multiprocessing.Pool() as pool:
        while remaining_sims > 0:
            cur_batch = min(batch_size, remaining_sims)
            params = []

            for i in range(cur_batch):
                p = (completed_sims + i, intruder_can_turn)
                params.append(p)
            
            results = pool.starmap(sim_single, params)
            print()
            
            for index, dist in enumerate(results):
                seed = completed_sims + index
                
                if dist < min_dist:
                    min_dist = dist
                    min_seed = seed

                if dist < collision_dist:
                    num_collisions += 1
                    collision_seeds.append(seed)

                    init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)
                    s = State(init_vec, init_velo[0], init_velo[1])

                    print(f"{num_collisions}. Collision (dist={round(dist, 2)}) with seed {seed}: {s}")

                if dist == np.inf:
                    num_rejected += 1

            # print progress
            completed_sims += cur_batch
            remaining_sims -= cur_batch

            frac = completed_sims / num_sims
            percent = round(100 * frac, 3)
            elapsed = time.perf_counter() - start
            total_estimate = (elapsed / frac)
            total_min = round(total_estimate / 60, 1)
            eta_estimate = total_estimate - elapsed
            eta_min = round(eta_estimate / 60, 1)
            
            print(f"{completed_sims}/{num_sims} ({percent}%) total estimate: {total_min}min, ETA: {eta_min} min, " + \
                f"rej: {num_rejected} ({round(100 * num_rejected / completed_sims, 6)}%), " + \
                f"col: {num_collisions} ({round(100 * num_collisions / completed_sims, 6)}%)")

    diff = time.perf_counter() - start
    ms_per_sim = round(1000 * diff / num_sims, 3)
    print(f"\nDid {num_sims} parallel sims in {round(diff, 1)} secs ({ms_per_sim}ms per sim)")
    print(f"Collision seeds: {collision_seeds}")

    print(f"Rejected {num_rejected} sims. {round(100 * num_rejected / num_sims, 6)}%")
    
    print(f"Collision in {num_collisions} sims. {round(100 * num_collisions / num_sims, 6)}%")

    d = round(min_dist, 1)
    print(f"\nSeed {min_seed} has min_dist {d}ft")

    # optional: do plot
    init_vec, cmd_list, init_velo = make_random_input(min_seed, intruder_can_turn=intruder_can_turn)
    s = State(init_vec, init_velo[0], init_velo[1], save_states=True)
    s.simulate(cmd_list)
    assert abs(s.min_dist - min_dist) < 1e-6, f"got min dist: {s.min_dist}, expected: {min_dist}"
    
    plot(s, save_mp4)

if __name__ == "__main__":
    main()
