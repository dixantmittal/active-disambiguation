import numpy as np

from string_utils import *

n_runs = 100


class Object(object):
    def __init__(self, caption):
        caption = clean_punctuation(caption)
        self.description = caption
        self.tokens = self.get_tokens(caption)

    def get_tokens(self, caption):
        return set(caption.split(' ')) - {'in', 'on', 'the', 'of', 'a', 'an'}

    def __str__(self):
        return self.description

    def __len__(self):
        return len(self.tokens)


KNOWLEDGE = [
    Object('a yellow cup on left'),
    Object('a yellow cup in middle'),
    Object('a yellow cup on right'),
    Object('a red cup on left'),
    Object('a red cup in middle'),
    Object('a red cup on right'),
    Object('a green cup on left'),
    Object('a green cup in middle'),
    Object('a green cup on right'),
    Object('a blue cup on left'),
    Object('a blue cup in middle'),
    Object('a blue cup on right')
]
WORLD = list(KNOWLEDGE)

ACTIONS = set()
for obj in KNOWLEDGE:
    ACTIONS.update(obj.tokens)

ACTIONS = list(ACTIONS)

OBSERVATIONS = ['yes', 'no']

n_objects = len(KNOWLEDGE)
n_actions = len(ACTIONS)
n_observations = len(OBSERVATIONS)


def random_scenario():
    global KNOWLEDGE, ACTIONS, OBSERVATIONS, n_objects, n_actions, n_observations

    scenario = set()

    i = np.minimum(np.maximum(np.random.randint(len(WORLD)), 3), 8)
    while i > 0:
        idx = np.random.randint(len(WORLD))
        if idx not in scenario: i -= 1
        scenario.add(idx)

    KNOWLEDGE = []
    for i in scenario:
        KNOWLEDGE.append(WORLD[i])

    semantic = set()
    spatial = set()
    for desc in KNOWLEDGE:
        semantic.add(desc[0])
        spatial.add(desc[1])

    ACTIONS = list(semantic.union(spatial))

    OBSERVATIONS = ['yes', 'no']

    n_objects = len(KNOWLEDGE)
    n_actions = len(ACTIONS)
    n_observations = len(OBSERVATIONS)
