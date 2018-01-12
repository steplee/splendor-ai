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
n-step Policy Iteration.

I will first implement 1-step (which is called Value Iteration).
I should also implement experience replay and see how it effects performance.
'''


class ValueIterationActor(BaseActor):
    def __init__(self, use_model=None, **model_params):
        self.name = "ValueIterationActor"

        if type(use_model) == type:
            self.model = use_model(*model_params)
        else:
            self.model = use_model

        ''' todo: finish this '''
        self.final_states_not_applied = []
        # We can randomly sample from this, which maps ints to history triplets, rather than continuing game.
        self.memory = {}
        # These are triplets from the history which are ready for a minibatch update.
        self.finished_games = []

    ' todo: implement experience replay '
    ' Return (status,record)'
    def act(self, record, training=True):
        next_state,values_var,action_id = self.sample_action_with_model(record.state, training)

        if type(values_var) == type(None) and action_id == -1:
            return ('fail',-1), None

        #if not next_state.verify_game_valid():
            #print("Invalid game")

        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,self,stat,record.state.active_player)

        if np.random.random()>.999 and type(values_var)!=type(None):
            #print('game_state',next_state.arr)
            print('vi.cross_entropy weights norm',sum(torch.norm(p) for p in self.model.parameters()).data.numpy()[0])
            print('vi action values',values_var.data.numpy()[:,0])

        if stat[0] == 'won':
            return stat, rec
        elif stat[0] == 'draw':
            return stat, rec

        return stat,rec


    ' return <next_state, values_var, act_id> '
    def sample_action_with_model(self, gstate, training=True):
        # Gather future 1-step states
        future_games = gstate.simulate_all_actions(gstate.active_player)
        future_games = [fg for fg in future_games if fg != None]

        ' Make sure we have atleast one good action '
        if len(future_games) == 0:
            print("vi all bad")
            return gstate, None, -1 # return old state

        gsts = [fg.get_game_features() for fg in future_games]
        csts = [fg.get_card_features_as_2d() for fg in future_games]
        fsts,gsts = np.array(gsts), np.array(csts)

        # Evaluate future states with the Value Function.
        values = self.model(fsts, gsts)

        # sample from it
        probs = np.copy(values.data.numpy()[:,0])

        ' If training, sample wrt probs. If testing, select max '
        if (not training):
            act_id = probs.argmax()
            #print(gstate.get_player_whole(0),'->',future_games[act_id].get_player_whole(1))
        else:
            probs = probs / sum(probs)
            act_id = np.random.choice(range(len(probs)), p=probs)


        return future_games[act_id], values, act_id



    def apply_reward(self, final_record):
        if len(self.final_states_not_applied) < 1:
            self.final_states_not_applied.append(final_record)
        else:
            self.final_states_not_applied.append(final_record)

            loss = None

            for record in self.final_states_not_applied:
                winner = record.status[1]

                ' These all run in reverse, we are following the linked list backward '
                value_vars = []
                selected_acts = []
                pids = []

                while record != None and record.prev:
                    assert( record.prev != record ) # loop?!

                    # Skip over a record if e.g. made by a RandomActor
                    if type(record.values_var) != type(None):
                        if record.actor == self:
                            value_vars.append(record.values_var)
                            selected_acts.append(record.act_id)
                            pids.append(record.pid)
                    record = record.prev

                if len(value_vars) == 0 or len(selected_acts) == 0:
                    continue


                # L2 Loss
                #selected_acts_ids = torch.LongTensor(selected_acts)
                selected_acts_v = torch.cat([vv[sa] for (vv,sa) in zip(value_vars,selected_acts)])
                targets = [10.0 if winner==pid else 1 for pid in pids]
                targets = ag.Variable(torch.FloatTensor(targets), requires_grad=False)
                weight = +1.0 if winner==pids[0] else .7
                if loss is None:
                    loss = torch.nn.functional.l1_loss(selected_acts_v, targets, reduce=True) * weight
                else: 
                    loss += torch.nn.functional.l1_loss(selected_acts_v, targets, reduce=True) * weight


            loss.backward()
            print("VI stepping with loss:",loss.data[0])
            loss = None
            self.model.opt.step()
            self.model.zero_grad()

            self.final_states_not_applied = []

        return True
