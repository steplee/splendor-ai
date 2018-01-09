import numpy as np
from numpy import random as npr

import random,sys
import itertools
from collections import namedtuple

#import samplers

from util import COIN_PICKUPS, one_hot, one_neg, LEVEL_PROBABILITIES, Record, Action, Actions

import torch
import torch.autograd as ag


from actors.BaseActor import BaseActor

'''
Random Actor. that's right, it acts randomly.
'''


class RandomActor(BaseActor):
    def __init__(self, use_model=None):
        self.name = "RandomActor"
        pass

    def act(self, record, training=True):

        next_state,values_var,action_id = self.sample_action_with_model(record.state)

        if type(values_var) == type(None) and action_id == -1:
            return ('fail',-1), None

        if not next_state.verify_game_valid():
            print("Invalid game")

        ' The random actor has no model. He puts None into the record '
        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,self,stat,record.state.active_player)

        if np.random.random()>.999 and type(values_var)!=type(None):
            #print('game_state',next_state.arr)
            print('action values',values_var.data.numpy()[:,0])

        if stat[0] == 'won':
            return stat, rec
        elif stat[0] == 'draw':
            return stat, rec

        return stat,rec


    ' return <next_state, values_var, act_id> '
    def sample_action_with_model(self, gstate, training=True):
        # Gather future 1-step states
        future_games = gstate.simulate_all_actions(gstate.active_player)
        if all( (fg == None for fg in future_games) ):
            print("all bad")
            return gstate, None, -1 # return old state


        # sample a state to go to
        good_choices = [i for (i,fgame) in enumerate(future_games) if fgame != None]
        act_id = np.random.choice(good_choices)

        return future_games[act_id], None, act_id



    def apply_reward(self, final_record):
        return True
