from game import Actions

import numpy as np
import game


'''
All sample functions return a 'trials' integer representing the # times we sampled
before getting a valid move. If we never get a valid move, return a tuple <'noop', argmax_score, 301>
meaning it should be **decreased**
'''

def random_sample(nscores,pid,game):
    act,act_id = None,None
    for trials in range(300):
        cand = int(np.random.choice(range(len(nscores)),1)[0])
        if game.action_is_valid_for_game(pid,cand):
            act_id = cand
            act = Actions[cand]
            break

    if not act:
            print(" BAD: performing a noop!")
            #print(nscores)
            return ('noop', np.argmax(scores), 301)

    return (act, act_id, trials)

def argmax_sample(nscores,pid,game):
    act,act_id = None,None
    for trials in range(len(nscores)):
        cand_id = int(nscores.argmax())
        cand = Actions[cand_id]
        if game.action_is_valid_for_game(pid,cand):
            act_id = cand_id
            act = cand
            break
        else:
            nscores[cand_id] = 0

    if not act or nscores[act_id] < .0000001:
            print(" BAD: performing a noop!")
            #print(nscores)
            return ('noop', np.argmax(nscores), 301)

    return (act, act_id, trials)

'''
Samples using scores as weighting of a multinoulli

returns an <act, act_id, trials>
'''
def proportional_sample(nscores, game, pid=None):
    act,act_id = None,None

    for trials in range(300):
        cand_id = int(np.random.choice(range(len(nscores)),1,p=nscores)[0])
        cand = Actions[cand_id]
        if (pid==None) or game.action_is_valid_for_game(pid,cand):
            act_id = cand_id
            act = cand
            break

    if not act:
            print(" BAD: performing a noop!")
            #print(nscores)
            return ('noop', np.argmax(nscores), 301)

    return (act, act_id, trials)


def epsilon_greedy_sample(nscores, pid, Epsilon, game):
    act,act_id = None,None
    eps = np.random.random()
    trials = 0

    if eps > Epsilon: 
        # greedy action, choose best that is valid
        while act is None or not game.action_is_valid_for_game(pid,act):
            trials += 1
            act_id = int(np.argmax(nscores))
            if nscores[act_id] == -1:
                break
            nscores[act_id] = -1
            act = Actions[act_id]

        if not act:
            return (act, act_id, trials)


    if act is None:
        # choose random
        trials = 0
        act,act_id = None,None
        while act is None or not game.action_is_valid_for_game(pid,act):
            act_id = int(np.random.choice(range(len(nscores)),1)[0])
            act = decode_act_id(act_id)
            trials += 1
            if trials >= 200:
                print(" BAD: performing a noop!")
                #print(nscores)
                return ('noop', np.argmax(nscores), 301)

    return (act, act_id, trials)
