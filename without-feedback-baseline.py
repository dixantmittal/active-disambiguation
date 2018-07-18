import numpy as np


def main():
    knowledge = np.array([["red cup", "left"],
                          ["red cup", "middle"],
                          ["blue cup", "right"],
                          ["blue cup", "left"],
                          ["blue cup", "middle"],
                          ["blue cup", "right"]])

    size = len(knowledge)

    # belief = np.array([1,5])
    belief = np.ones(size)
    belief = belief / belief.sum(keepdims = True)

    print("Picks: ", knowledge[belief.argmax()])


if __name__ == '__main__':
    main()
