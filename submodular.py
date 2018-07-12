from commons import *


def plan(belief, n_questions = 3):
    optimal_actions = []
    while n_questions > 0:
        information_gain = np.zeros(n_actions)
        for _action in range(n_actions):
            information_gain[_action] = conditional_entropy(_action, optimal_actions, belief)

        n_questions -= 1
        optimal_actions.append(information_gain.argmax())

    return optimal_actions


if __name__ == '__main__':

    for _ in range(n_runs):
        belief = np.ones(n_objects)
        belief = belief / belief.sum(keepdims = True)
        try:
            while True:
                actions = plan(belief, 1)
                action = actions[0]

                for action in actions:
                    if belief.max() > 0.7:
                        producer.send('active-disambiguation-questions',
                                      ('Picks up ' + KNOWLEDGE[belief.argmax()][0] + ' on ' + KNOWLEDGE[belief.argmax()][1]).encode('utf-8'))
                        producer.flush()
                        print('Picks up ' + KNOWLEDGE[belief.argmax()][0] + ' on ' + KNOWLEDGE[belief.argmax()][1])
                        raise KeyError

                    producer.send('active-disambiguation-questions', ('Did you mean the ' + ACTIONS[action] + '?').encode('utf-8'))
                    producer.flush()

                    for reply in consumer:
                        actual_observation = reply.value.decode('utf-8')
                        break

                    belief = belief_update(belief, ACTIONS[action], actual_observation)
                    if actual_observation is 'yes':
                        break

        except KeyError:
            pass
        except KeyboardInterrupt:
            break

    consumer.commit()
