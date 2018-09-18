import numpy as np

import environment as env


def plan(belief, n_questions = 3):
    p_mass_removed = np.zeros(env.n_actions)
    for a, action in enumerate(env.ACTIONS):
        p_yes = np.zeros(env.n_objects)
        for i, obj in enumerate(env.KNOWLEDGE):
            p_yes[i] = belief[i] * (1 if action in obj.tokens else 0)

        p_mass_removed[a] = 2 * p_yes.sum() * (1 - p_yes.sum())

    return p_mass_removed.argmax()
