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

        self.finished_games = []
        self.final_states_not_applied = []
        self.invalid_games = []

    ' Return (status,record)'
    def act(self, record, training=True):
        next_state,values_var,action_id,acts = self.sample_action_with_model(record.state, training)

        if type(values_var) == type(None) and action_id == -1:
            return ('fail',-1), None

        #if not next_state.verify_game_valid():
            #print("Invalid game")

        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,self,stat,record.state.active_player)

        if np.random.random()>.998 and type(values_var)!=type(None):
            #print('game_state',next_state.arr)
            print('vi.cross_entropy weights norm',sum(torch.norm(p) for p in self.model.parameters()).data.numpy()[0])
            print('vi action values')
            for i,v in zip(acts, values_var.data.numpy()[:,0]):
                print(Actions[i],':',v)

        if stat[0] == 'won':
            return stat, rec
        elif stat[0] == 'draw':
            return stat, rec

        return stat,rec


    ' return <next_state, values_var, act_id, valid_action_ids> '
    def sample_action_with_model(self, gstate, training=True):
        # Gather future 1-step states
        future_games = gstate.simulate_all_actions(gstate.active_player)

        # randomly penalize some invalid states
        invalid_games = [fg[1] for fg in future_games if fg[0] == False and np.random.random()>0]
        bad_gsts = [fg.get_game_features() for fg in invalid_games]
        bad_csts = [fg.get_card_features_as_2d() for fg in invalid_games]
        bad_gsts,bad_csts = np.array(bad_gsts), np.array(bad_csts)
        if len(invalid_games) > 0:
            values = self.model(bad_gsts, bad_csts)
            invalid_records = list(zip(invalid_games,values))
            self.invalid_games += invalid_records



        actions = [i for i,fg in enumerate(future_games) if fg[0] == True]
        future_games = [fg[1] for fg in future_games if fg[0] == True]

        ' Make sure we have atleast one good action '
        if len(future_games) == 0:
            print("vi all bad")
            return gstate, None, -1, [] # return old state

        gsts = [fg.get_game_features() for fg in future_games]
        csts = [fg.get_card_features_as_2d() for fg in future_games]
        gsts,csts = np.array(gsts), np.array(csts)

        # Evaluate future states with the Value Function.
        values = self.model(gsts, csts)

        # sample from it
        probs = np.copy(values.data.numpy()[:,0])

        ' If training, sample wrt probs. If testing, select max '
        if training:
            probs = probs / sum(probs)
            act_id = np.random.choice(range(len(probs)), p=probs)
        else:
            act_id = probs.argmax()
            #print(gstate.get_player_whole(0),'->',future_games[act_id].get_player_whole(1))


        return future_games[act_id], values, act_id, actions



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
                targets = [1.0 if winner==pid else 0.2 for pid in pids]
                targets = ag.Variable(torch.FloatTensor(targets), requires_grad=False)
                weight = +1.0 if winner==pids[0] else .7
                if loss is None:
                    #loss = torch.nn.functional.l1_loss(selected_acts_v, targets, reduce=True) * weight
                    loss = torch.nn.functional.mse_loss(selected_acts_v, targets, reduce=True) * weight
                else: 
                    #loss += torch.nn.functional.l1_loss(selected_acts_v, targets, reduce=True) * weight
                    loss += torch.nn.functional.mse_loss(selected_acts_v, targets, reduce=True) * weight

            bad_loss = None
            for (game,scores) in self.invalid_games:
                targets = torch.Tensor([0.0 for _ in range(len(scores))])
                targets = ag.Variable(targets,requires_grad=False)
                if bad_loss is None:
                    bad_loss = torch.nn.functional.mse_loss(scores, targets)
                else:
                    bad_loss += torch.nn.functional.mse_loss(scores, targets)

            if type(bad_loss) != type(None):
                bad_loss *= .0001
                bad_loss.backward()
                self.invalid_games = []


            loss.backward()
            print("VI stepping with loss:",loss.data[0])
            loss = None
            self.model.opt.step()
            self.model.zero_grad()

            self.final_states_not_applied = []

        return True
