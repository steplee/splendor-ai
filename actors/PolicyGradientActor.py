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
Poliy Gradient Actor.

Direct Softmax over actions, given current state.

TODO: implement experience replay
'''


class PolicyGradientActor(BaseActor):
    def __init__(self, use_model=None, **model_params):

        if type(use_model) == type:
            self.model = use_model(*model_params)
        else:
            self.model = use_model

        self.final_states_not_applied = []

    ' todo: implement experience replay '
    ' Return (status,record)'
    def act(self, record, training=True):
        next_state,values_var,action_id = self.sample_action_with_model(record.state, model,training)

        if type(values_var) == type(None) and action_id == -1:
            return ('fail',-1), None

        if not next_state.verify_game_valid():
            print("Invalid game")

        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,self,stat,record.state.active_player)

        if np.random.random()>.999 and type(values_var)!=type(None):
            #print('game_state',next_state.arr)
            print('weights norm',sum(torch.norm(p) for p in self.model.parameters()).data.numpy()[0])
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

        # TODO only compute what works, need to un-hardcode 28x batching in the model code

        # Evaluate a distribution over actions.
        values = self.model(gstate.get_game_features, gstate.get_card_features())

        # sample from it
        good_choices = [i for (i,fgame) in enumerate(future_games) if fgame != None]
        probs = np.copy(values.data.numpy()[:,0])

        if training:
            probs = probs[good_choices]
            probs = probs / sum(probs)
            act_id = np.random.choice(good_choices, p=probs)
        else:
            good_choices = set(good_choices) # todo optimize...
            act_id = probs.argmax()
            while act_id not in good_choices:
                probs[act_id] = -.01
                act_id = probs.argmax()

        # Save our pytorch Variable, action taken, and model used
        #self.history.append((values, act_id, model))

        return future_games[act_id], values, act_id



    def apply_reward(self, final_record):
        if len(self.final_states_not_applied) < 10:
            self.final_states_not_applied.append(final_record)
        else:
            winner = final_record.status[1]
            assert(type(winner) == int and winner>= 0 and winner < 4)


            loss = None

            for record in self.final_states_not_applied:

                ' These all run in reverse, we are following the linked list backward '
                value_vars = []
                selected_acts = []
                weights = []
                pids = []

                while record != None and record.prev:
                    assert( record.prev != record ) # loop?!

                    # Skip over a record if e.g. made by a RandomActor
                    if type(record.values_var) != type(None):
                        if record.actor == self:
                            value_vars.append(record.values_var)
                            selected_acts.append(record.act_id)
                            weights.append(1.0 if record.state.active_player == winner else -.33)
                            pids.append(record.pid)
                    record = record.prev

                if len(value_vars) == 0 or len(selected_acts) == 0:
                    continue


                '''
                Loss: Softmax + xent
                '''

                # XEnt loss
                #targets = ag.Variable(torch.LongTensor(acts), requires_grad=False)
                #loss = torch.nn.functional(value_vars, targets, reduce=False)
                ## If we lose the game use the *negative of xent* as objective.
                #weights = ag.Variable(torch.FloatTensor(weights), requires_grad=False)
                #loss = torch.average(loss * weights)



                # L2 Loss
                #selected_acts_ids = torch.LongTensor(selected_acts)
                selected_acts_v = torch.cat([vv[sa] for (vv,sa) in zip(value_vars,selected_acts)])
                targets = [30.0 if winner==pid else 0.01 for pid in pids]
                targets = ag.Variable(torch.FloatTensor(targets), requires_grad=False)
                if loss is None:
                    loss = torch.nn.functional.mse_loss(selected_acts_v, targets, reduce=True)
                else:
                    loss += torch.nn.functional.mse_loss(selected_acts_v, targets, reduce=True)


            loss.backward()
            print("PG Actor stepping with loss:",loss.data[0])
            self.model.opt.step()
            self.model.zero_grad()

            self.final_states_not_applied = []

        return True

