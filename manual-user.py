import numpy as np

import adaptive_submodular as ads
import commons as coms
import environment as env

np.set_printoptions(precision = 2, suppress = True)

belief = np.ones(env.n_objects)
belief = belief / belief.sum(keepdims = True)

ref_exp = input('Which object do you want?\n>')
belief = coms.belief_update(belief, '', ref_exp)

nq = 0
while belief.max() < 0.7:
    print(belief)
    question = ads.plan(belief)
    print('Did you mean the ' + env.ACTIONS[question] + ' object?')
    actual_observation = input('> ')
    belief = coms.belief_update(belief, env.ACTIONS[question], actual_observation)
    nq += 1

print('Picks up: ', env.KNOWLEDGE[belief.argmax()])
