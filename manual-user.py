import argparse

import numpy as np

import adaptive_submodular as ads
import commons as coms
import environment as env
import greedy as gd
import pomdp as pd
import submodular as sm

np.set_printoptions(precision = 2, suppress = True)

MODE = {
    'pomdp': pd.plan,
    'greedy': gd.plan,
    'submodular': sm.plan,
    'adasub': ads.plan
}

parser = argparse.ArgumentParser(description = 'Mode to simulate: ')
parser.add_argument('--planner',
                    dest = 'planner',
                    default = 'adasub')

args = parser.parse_args()

belief = np.ones(env.n_objects)
belief = belief / belief.sum(keepdims = True)

nq = 0
while belief.max() < 0.8:
    question = MODE[args.planner](belief)
    print('Did you mean the ' + env.ACTIONS[question] + ' object?')
    actual_observation = input('> ')
    belief = coms.belief_update(belief, env.ACTIONS[question], actual_observation)
    nq += 1

print('Picks up: ', env.KNOWLEDGE[belief.argmax()])
