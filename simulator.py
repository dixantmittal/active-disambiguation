import argparse
import itertools
import random
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import adaptive_submodular as ads
import commons as coms
import environment as env
import pomdp as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

MODE = {
    'pomdp': pd.plan,
    'adaptive-submodular': ads.plan
}


def get_reply(question, intention = None):
    return 'yes' if question in env.KNOWLEDGE[intention].tokens else 'no'


def simulation(ref_exp, intention, planner):
    belief = np.ones(env.n_objects)
    belief = belief / belief.sum(keepdims = True)
    belief = coms.belief_update(belief, '', ref_exp)

    nq = 0
    while belief.max() < 0.75:
        question = env.ACTIONS[MODE[planner](belief)]
        reply = get_reply(question, intention)
        belief = coms.belief_update(belief, question, reply)
        nq += 1
    return nq


def simulate_point(ref_exp, intention):
    belief = np.ones(env.n_objects)
    belief = belief / belief.sum(keepdims = True)
    belief = coms.belief_update(belief, '', ref_exp)

    nq = 1
    point = belief.argmax()
    while point != intention:
        belief[point] *= 1 if point == intention else 0
        belief = belief / belief.sum(keepdims = True)
        point = belief.argmax()
        nq += 1

    return nq


def start_experiment(n_scenarios = 5, n_runs = 20):
    pool = ThreadPool(100)

    t_adasub = []
    t_pomdp = []
    t_point = []

    r_adasub = []
    r_pomdp = []
    r_point = []

    for _ in tqdm(range(n_scenarios)):
        env.random_scenario()
        intentions = np.random.choice(a = env.n_objects, size = n_runs)
        ref_exp = []
        for i in range(n_runs):
            ref_exp += random.sample(env.KNOWLEDGE[intentions[i]].tokens, 1)

        # measure performance for Adaptive Submodular approach
        start = time()
        n_questions_adasub = pool.starmap(simulation, zip(ref_exp, intentions, itertools.repeat('adaptive-submodular')))
        t_adasub.append((time() - start) / n_runs)
        r_adasub.append(n_questions_adasub)

        start = time()
        n_questions_pomdp = pool.starmap(simulation, zip(ref_exp, intentions, itertools.repeat('pomdp')))
        t_pomdp.append((time() - start) / n_runs)
        r_pomdp.append(n_questions_pomdp)

        start = time()
        n_questions_point = pool.starmap(simulate_point, zip(ref_exp, intentions))
        t_point.append((time() - start) / n_runs)
        r_point.append(n_questions_point)

    print(t_adasub)
    print(t_pomdp)
    print(t_point)

    plt.figure(dpi = 600)

    x = np.arange(21, n_scenarios + 21)

    y = np.array(r_adasub).mean(axis = 1)
    e = np.array(r_adasub).std(axis = 1)
    plt.errorbar(x - 0.2, y, e, linestyle = 'None', linewidth = 0.5, marker = 'o', markersize = 3, capsize = 3, label = 'Adaptive Submodularity')

    y = np.array(r_pomdp).mean(axis = 1)
    e = np.array(r_pomdp).std(axis = 1)
    plt.errorbar(x, y, e, linestyle = 'None', linewidth = 0.5, marker = 'o', markersize = 3, capsize = 3, label = 'POMDP')

    y = np.array(r_point).mean(axis = 1)
    e = np.array(r_point).std(axis = 1)
    plt.errorbar(x + 0.2, y, e, linestyle = 'None', linewidth = 0.5, marker = 'o', markersize = 3, capsize = 3, label = 'Greedy Pointing')

    plt.xticks(x)
    plt.xlabel('Scenario ID')
    plt.ylabel('Number of Questions (lower is better)')
    plt.title('Simulation Results')

    plt.legend()
    plt.savefig('plots/' + str(time()) + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs',
                        dest = 'n_runs',
                        default = '10',
                        help = 'number of runs')
    parser.add_argument('--n_scenarios',
                        dest = 'n_scenarios',
                        default = '5',
                        help = 'number of scenarios')
    args = parser.parse_args()

    start_experiment(int(args.n_scenarios), int(args.n_runs))
