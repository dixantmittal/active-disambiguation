import numpy as np

from environment import *

np.set_printoptions(precision = 2, suppress = True)


# P(z|s,a)
def p_obs(z, s, a):
    return 0.01 if (a in s and z == 'no') or (a not in s and z == 'yes') else 0.99


# P(z|b,a)
def obs_likelihood(belief, action, obs):
    likelihood = np.zeros(len(belief))
    for i, obj in enumerate(KNOWLEDGE):
        likelihood[i] = p_obs(obs, obj[0] + ' ' + obj[1], action) * belief[i]
    return likelihood


# P(s|b,a,z)
def belief_update(belief, action, obs):
    updated_belief = obs_likelihood(belief, action, obs)
    return updated_belief / updated_belief.sum(keepdims = True)


# P(z1, z2, z3) = \sum_s P(z1|s) * P(z2|s) * P(z3|s)
# variable =                    observations with proper index
# conditional_distributions =   array containing distributions
def joint_probability(variables, belief):
    probability = 0
    for _object in range(n_objects):
        conditional = 1
        for observation in variables:
            conditional = conditional * DISTRIBUTIONS[_object, observation]

        probability = probability + conditional * belief[_object]

    return probability


# H(X|U) = - \sum_u P(u) \sum_x P(x|u) log P(x|u)
def conditional_entropy(conditional_variable, observed_variables = [], belief = None):
    _entropy = 0

    for i in range(2 ** len(observed_variables)):
        observations = []
        for variable in observed_variables:
            observations.append(variable * 2 + i % 2)
            i = i // 2

        j_p = np.zeros(n_observations)

        j_p[0] = joint_probability(observations + [conditional_variable * 2], belief)
        j_p[1] = joint_probability(observations + [conditional_variable * 2 + 1], belief)

        _entropy = _entropy - np.dot(j_p, (np.log2(j_p) - np.log2(j_p.sum())))

    return _entropy


def entropy(p):
    return -np.dot(p, np.log2(p))


def entropy_diff(p1, p2):
    return entropy(p1) - entropy(p2)


def init_distributions():
    dist = np.zeros((len(KNOWLEDGE), len(ACTIONS), len(OBSERVATIONS)))
    for i, _object in enumerate(KNOWLEDGE):
        for j, action in enumerate(ACTIONS):
            for k, observation in enumerate(OBSERVATIONS):
                dist[i, j, k] = p_obs(observation, _object, action)

    return dist


DISTRIBUTIONS = init_distributions().reshape((n_objects, -1))
