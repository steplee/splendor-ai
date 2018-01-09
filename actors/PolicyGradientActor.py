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
        self.name = "PolicyGradientActor"

        if type(use_model) == type:
            self.model = use_model(*model_params)
        else:
            self.model = use_model

        self.final_states_not_applied = []

    ' todo: implement experience replay '
    ' Return (status,record)'
    def act(self, record, training=True):
        next_state,values_var,action_id = self.sample_action_with_model(record.state, training)

        if type(values_var) == type(None) and action_id == -1:
            return ('fail',-1), None

        if not next_state.verify_game_valid():
            print("Invalid game")

        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,self,stat,record.state.active_player)

        if np.random.random()>.999 and type(values_var)!=type(None):
            #print('game_state',next_state.arr)
            print('pg weights norm',sum(torch.norm(p) for p in self.model.parameters()).data.numpy()[0])
            print('pg scores',values_var.data.numpy()[0])

        if stat[0] == 'won':
            return stat, rec
        elif stat[0] == 'draw':
            return stat, rec

        return stat,rec


    ' return <next_state, values_var, act_id> '
    def sample_action_with_model(self, gstate, training=True):
        future_games = gstate.simulate_all_actions(gstate.active_player)
        if all( (fg == None for fg in future_games) ):
            print("pg all bad")
            return gstate, None, -1

        # Evaluate a distribution over actions.
        fst =  [gstate.get_game_features()]
        csts = [gstate.get_card_features_as_2d()]
        values = self.model(fst,csts)

        # sample from it
        good_choices = [i for (i,fgame) in enumerate(future_games) if fgame != None]
        probs = np.copy(values.data.numpy()[0])

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
                            selected_acts.append(int(record.act_id))
                            weights.append(1.0 if record.state.active_player == winner else -.33)
                            pids.append(record.pid)
                    record = record.prev

                if len(value_vars) == 0 or len(selected_acts) == 0:
                    continue


                '''
                Loss: Softmax + xent
                '''

                # XEnt loss
                weights = ag.Variable(torch.FloatTensor(weights), requires_grad=False)
                scores = torch.cat(value_vars)
                targets = ag.Variable(torch.LongTensor(selected_acts), requires_grad=False)

                local_loss = torch.nn.functional.cross_entropy(scores, targets, reduce=False)
                local_loss = torch.sum(local_loss * weights)

                if loss is None:
                    loss = local_loss
                else:
                    loss += local_loss



            loss.backward()
            print("PG Actor stepping with loss:",loss.data[0])
            self.model.opt.step()
            self.model.zero_grad()

            self.final_states_not_applied = []

        return True

