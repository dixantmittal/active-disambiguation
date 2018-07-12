import argparse
from environment import *
from kafka import KafkaConsumer, KafkaProducer
import numpy as np

consumer = KafkaConsumer('active-disambiguation-questions', group_id = 'alpha')
producer = KafkaProducer(bootstrap_servers = 'localhost:9092')


def read():
    for question in consumer:
        return question.value.decode('utf-8')
    consumer.commit_async()


def reply(message):
    producer.send('active-disambiguation-replies', message.encode('utf-8'))
    producer.flush()


def simulate_pomdp():
    pass


def simulate_greedy():
    pass


def prepare_reply(question, intention):
    return 'yes' if KNOWLEDGE[intention][0] in question or KNOWLEDGE[intention][1] in question else 'no'


def simulate_submodular():
    intention = np.random.randint(len(KNOWLEDGE))
    n_questions = 0
    print('Intention :', KNOWLEDGE[intention])
    print()
    while True:
        question = read()
        print(question)

        if 'Picks up' in question:
            break

        to_reply = prepare_reply(question, intention)
        print(to_reply)
        reply(to_reply)
        n_questions += 1

    print('\nQuestions asked :', n_questions, '\n')
    return n_questions


def user():
    while True:
        question = read()
        print(question)

        if 'Picks up' in question:
            break

        reply(input())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Mode to simulate: ')
    parser.add_argument('--mode', default = 'user',
                        help = 'pomdp greedy submodular [user]')

    args = parser.parse_args()

    if args.mode is 'pomdp':
        simulate_pomdp()
    elif args.mode is 'greedy':
        simulate_greedy()
    elif args.mode is 'submodular':
        simulate_submodular()
    else:
        n_q = 0
        for i in range(n_runs):
            n_q += simulate_submodular()

        print('Average questions: ', n_q / n_runs)
