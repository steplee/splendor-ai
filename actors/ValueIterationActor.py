import numpy as np
from numpy import random as npr

import random,sys
import itertools
from collections import namedtuple

#import samplers

from util import COIN_PICKUPS, one_hot, one_neg, LEVEL_PROBABILITIES, Record, Action, Actions

import torch
import torch.autograd as ag

'''
Some actors may need a history, some may need an episode bank.

Either way, after the actor will do what he needs when 'act' is called, he will return a state to continue upon.
This could be the next state continuing the same game, or a replayed experience from his memory.

We check after every action if the player has won. We collect all finished games and apply
when hitting the minibatch size.

'''


class Actor(object):
    def __init__(self):
        pass

    ' Must some state to continue with (eg. next state, random episode from some history...) '
    def act(self, gstate):
        pass

'''
n-step Policy Iteration.

I will first implement 1-step (which is called Value Iteration).
I should also implement experience replay and see how it effects performance.
'''


class ValueIterationActor(object):
    def __init__(self):
        # <ag.Variable, action_id, model>
        # We need to store the model in case we have multiple models in one game.
        self.history = []

        ''' todo: finish this '''
        # We can randomly sample from this, which maps ints to history triplets, rather than continuing game.
        self.memory = {}
        # These are triplets from the history which are ready for a minibatch update.
        self.finished_games = []

    ' todo: implement experience replay '
    ' Return (status,record)'
    def act(self, record, model):
        next_state,values_var,action_id = self.sample_action_with_model(record.state, model)

        if not next_state.verify_game_valid():
            print("Invalid game")

        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,model,stat,record.state.active_player)

        if np.random.random()>.99:
            print(stat)
            print(next_state.arr)
            print([next_state.get_player_points(i) for i in range(4)])

        if stat[0] == 'won':
            return stat, rec
        elif stat[0] == 'draw':
            return stat, rec

        return stat,rec


    ' return <next_state, values_var, act_id> '
    def sample_action_with_model(self, gstate, model):
        # Gather future 1-step states
        future_games = gstate.simulate_all_actions(gstate.active_player)
        if all( (fg == None for fg in future_games) ):
            return gstate, -1

        gsts = [fg.get_game_features() if fg != None else np.zeros(62) for fg in future_games]
        csts = [fg.get_card_features_as_2d() if fg != None else np.zeros([12,13]) for fg in future_games]
        fsts,gsts = np.array(gsts), np.array(csts)

        # Evaluate future states with the Value Function.
        values = model(fsts, gsts)

        # sample from it
        #np_values = values.data.numpy()
        probs = np.copy(values.data.numpy()[:,0])
        probs = probs / sum(probs)

        #print(probs)

        act_id = None
        trials = 0
        while act_id is None or future_games[act_id] != None and trials < 100:
            act_id = np.random.choice(range(len(Actions)), p=probs)
            trials += 1

        if act_id is None or future_games[act_id] is None:
            choices = [i for (i,fgame) in enumerate(future_games) if fgame != None]
            if len(choices) == 0:
                print("NO VALID MOVES!")
                return gstate, -1

            act_id = np.random.choice(choices)
            #print('warning, actions were mostly invalid, taking',act_id)

        #for i,p in enumerate(probs):
            #print(i,'->',p)
        '''
        print(sum(probs),'Taking',act_id)
        print(probs.shape)
        '''

        # Save our pytorch Variable, action taken, and model used
        self.history.append((values, act_id, model))

        return future_games[act_id], values, act_id



    def apply_reward(self, final_record):
        ' Currently, I am batching per game-ending '
        winner = final_record.status[1]
        assert(type(winner) == int and winner>= 0 and winner < 4)

        models = set()
        value_vars = []
        selected_acts = []
        weights = []
        pids = []

        record = final_record
        while record != None and record.prev:
            value_vars.append(record.values_var)
            selected_acts.append(record.act_id)
            weights.append(1.0 if record.state.active_player == winner else -.33)
            pids.append(record.pid)
            if record.model not in models:
                models.add(record.model)
                record.model.zero_grad()
            record = record.prev

        # XEnt loss
        #targets = ag.Variable(torch.LongTensor(acts), requires_grad=False)
        #loss = torch.nn.functional(value_vars, targets, reduce=False)
        ## If we lose the game use the *negative of xent* as objective.
        #weights = ag.Variable(torch.FloatTensor(weights), requires_grad=False)
        #loss = torch.average(loss * weights)

        # L2 Loss
        #selected_acts_ids = torch.LongTensor(selected_acts)
        print(len(selected_acts))
        selected_acts = torch.cat([vv[sa] for (vv,sa) in zip(value_vars,selected_acts)])
        targets = [30.0 if winner==pid else 0.01 for pid in pids]
        targets = ag.Variable(torch.FloatTensor(targets), requires_grad=False)
        loss = torch.nn.functional.mse_loss(selected_acts, targets, reduce=True)

        print("Loss:",loss.data[0])

        loss.backward()
        for model in models:
            print("STEPPING")
            model.opt.step()

        return True
