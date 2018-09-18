import numpy as np

import environment as env
from string_utils import *

np.set_printoptions(precision = 2, suppress = True)


# utils functions
def cosine(u, v):
    return np.dot(u, np.transpose(v)) / (np.linalg.norm(u, axis = 1, keepdims = True) * np.linalg.norm(v, axis = 1, keepdims = True))


def unigram_model(sentence):
    p_words = np.ones(env.n_objects)
    sentence = to_set(sentence, ' ')
    for i, obj in enumerate(env.KNOWLEDGE):
        common_words = len(obj.tokens.intersection(sentence))
        p_words[i] = len(obj) ** (-common_words) * (1e3 * len(obj)) ** (common_words - len(sentence))
    return p_words / p_words.sum(keepdims = True)


def sentence_similarity_score(sentence):
    return unigram_model(sentence)


# P(z|s,a)
def p_obs(z, s, a):
    return 1e-5 if (a in s.tokens and z in no_list) or (a not in s.tokens and z in yes_list) else 1 - 1e-5


def split_sentence(observation):
    observation = clean_punctuation(observation)

    response = ''
    description = observation
    for word in observation.split(' '):
        if word in (no_list + yes_list):
            response = word
            description = description.replace(word, '')
            break

    return response, description


# P(z|b,a)
def obs_likelihood(belief, action, observation):
    response, description = split_sentence(observation)

    likelihood = np.ones(env.n_objects)
    if response is not '':
        for i, obj in enumerate(env.KNOWLEDGE):
            likelihood[i] = p_obs(response, obj, action)

    if len(description) > 0:
        likelihood = likelihood * sentence_similarity_score(description)
    return likelihood * belief


# P(s|b,a,z)
def belief_update(belief, action, obs):
    updated_belief = obs_likelihood(belief, action, obs)
    return updated_belief / updated_belief.sum(keepdims = True)
