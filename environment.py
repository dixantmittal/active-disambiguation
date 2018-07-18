import numpy as np

n_runs = 100

KNOWLEDGE = WORLD = [
    ["yellow cup", "left"],
    ["yellow cup", "middle"],
    ["yellow cup", "right"],
    ["red cup", "left"],
    ["red cup", "middle"],
    ["red cup", "right"],
    ["green cup", "left"],
    ["green cup", "middle"],
    ["green cup", "right"],
    ["blue cup", "left"],
    ["blue cup", "middle"],
    ["blue cup", "right"]
]

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


def random_scenario():
    global KNOWLEDGE, ACTIONS, OBSERVATIONS, n_objects, n_actions, n_observations

    scenario = set()

    i = np.maximum(np.random.randint(len(WORLD)), 3)
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
