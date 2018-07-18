import numpy as np

import commons as coms
import environment as env


def plan(belief):
    information_gain = np.zeros(env.n_actions)
    for action in range(env.n_actions):
        for obs in env.OBSERVATIONS:
            updated_belief = coms.belief_update(belief, env.ACTIONS[action], obs)
            information_gain[action] = information_gain[action] + coms.entropy_diff(belief, updated_belief) * coms.obs_likelihood(belief,
                                                                                                                                  env.ACTIONS[action],
                                                                                                                                  obs).sum()
    return information_gain.argmax()
