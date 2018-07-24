import numpy as np

import commons as coms
import environment as env


# calculate H(S|A)
def conditional_entropy(prior, likelihood = None):
    if likelihood is None:
        likelihood = np.ones(env.n_objects)

    joint_distribution = prior * likelihood
    conditional_distribution = joint_distribution / joint_distribution.sum(keepdims = True)

    entropy = joint_distribution * np.log2(conditional_distribution)

    return -entropy.sum()


# calculate E[H(S|A)-H(S|AUZ)]
def expected_information_gain(prior, likelihood = None, observed = None):
    _conditional_entropy = np.zeros(env.n_observations)

    # p_obs = P(z|A)
    p_obs = ((prior * likelihood).reshape(-1, 1) * coms.DISTRIBUTIONS[:, observed * 2: observed * 2 + 2]).sum(axis = 0)
    p_obs = p_obs / p_obs.sum(keepdims = True)

    for obs in range(env.n_observations):
        _conditional_entropy[obs] = conditional_entropy(prior, likelihood * coms.DISTRIBUTIONS[:, observed * 2 + obs])

    entropy = conditional_entropy(prior, likelihood) - np.dot(p_obs, _conditional_entropy)

    return entropy


# builds a policy tree
def find_policy(prior, likelihood = None, observed = set(), steps_left = 3):
    if steps_left == 0:
        return None
    if likelihood is None:
        likelihood = np.ones(env.n_objects)

    _expected_information_gain = np.zeros(env.n_actions)
    for node in range(env.n_actions):
        if node in observed:
            continue
        _expected_information_gain[node] = expected_information_gain(prior, likelihood, node)

    best_node = _expected_information_gain.argmax()
    observed.add(best_node)

    node = {}
    node['optimal_value'] = best_node
    node['expected_information_gain'] = _expected_information_gain[best_node]
    node['yes'] = find_policy(prior, likelihood * coms.DISTRIBUTIONS[:, best_node * 2], set(observed), steps_left - 1)
    node['no'] = find_policy(prior, likelihood * coms.DISTRIBUTIONS[:, best_node * 2 + 1], set(observed), steps_left - 1)

    return node


def plan(belief, n_questions = 3):
    policy_tree = find_policy(belief, np.ones(env.n_objects), set(), n_questions)
    return policy_tree['optimal_value']
