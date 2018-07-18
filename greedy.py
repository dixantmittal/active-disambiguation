from commons import *


def plan(belief):
    information_gain = np.zeros(n_actions)
    for action in range(n_actions):
        for obs in OBSERVATIONS:
            updated_belief = belief_update(belief, ACTIONS[action], obs)
            information_gain[action] = information_gain[action] + entropy_diff(belief, updated_belief) * obs_likelihood(belief, ACTIONS[action],
                                                                                                                        obs).sum()
    return information_gain.argmax()
