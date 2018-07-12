n_runs = 100

KNOWLEDGE = [
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
