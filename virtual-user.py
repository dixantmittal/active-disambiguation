import argparse
import itertools
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import adaptive_submodular as ads
import commons as coms
import environment as env
import greedy as gd
import pomdp as pd
import submodular as sm

MODE = {
    'pomdp': pd.plan,
    'greedy': gd.plan,
    'submodular': sm.plan,
    'adaptive-submodular': ads.plan
}


def get_reply(robot_says, intention = None, simulate = True):
    if simulate:
        return 'yes' if env.KNOWLEDGE[intention][0] in robot_says or env.KNOWLEDGE[intention][1] in robot_says else 'no'
    else:
        return input('A: ')


def simulation(intention, planner):
    belief = np.ones(env.n_objects)
    belief = belief / belief.sum(keepdims = True)

    nq = 0
    while belief.max() < 0.8:
        question = MODE[planner](belief)
        robot_says = 'Did you mean the ' + env.ACTIONS[question] + '?'
        actual_observation = get_reply(robot_says = robot_says, intention = intention, simulate = True)
        belief = coms.belief_update(belief, env.ACTIONS[question], actual_observation)
        nq += 1

    return nq


def start_experiment(n_runs = 5):
    pool = ThreadPool(100)
    for _ in range(1, n_runs + 1):
        env.random_scenario()
        coms.reinit_distributions()

        print("######### RUN:", _, "#########")

        intentions = np.random.choice(a = env.n_objects, size = env.n_objects * 5)

        nq_pomdp = pool.starmap(simulation, zip(intentions, itertools.repeat('pomdp')))
        nq_greedy = pool.starmap(simulation, zip(intentions, itertools.repeat('greedy')))
        nq_sub = pool.starmap(simulation, zip(intentions, itertools.repeat('submodular')))
        nq_adasub = pool.starmap(simulation, zip(intentions, itertools.repeat('adaptive-submodular')))

        nq_sub = nq_sub.mean()
        nq_pomdp = nq_pomdp.mean()
        nq_greedy = nq_greedy.mean()
        nq_adasub = nq_adasub.mean()

        print(env.KNOWLEDGE)
        print(nq_pomdp, '\t', nq_greedy, '\t', nq_sub, '\t', nq_adasub, end = '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Mode to simulate: ')
    parser.add_argument('--n_runs',
                        dest = 'n_runs',
                        default = '5',
                        help = 'number of runs')
    args = parser.parse_args()

    start_experiment(int(args.n_runs))
