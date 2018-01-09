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
    def act(self, gstate, model):
        pass

    def maybe_print(self):
        raise Exception('implement')

'''
n-step Policy Iteration.

I will first implement 1-step (which is called Value Iteration).
I should also implement experience replay and see how it effects performance.
'''


class ValueIterationActor(Actor):
    def __init__(self):
        # <ag.Variable, action_id, model>
        # We need to store the model in case we have multiple models in one game.
        self.history = []

        ''' todo: finish this '''
        self.final_states_not_applied = []
        # We can randomly sample from this, which maps ints to history triplets, rather than continuing game.
        self.memory = {}
        # These are triplets from the history which are ready for a minibatch update.
        self.finished_games = []

    ' todo: implement experience replay '
    ' Return (status,record)'
    def act(self, record, model, training=True):
        next_state,values_var,action_id = self.sample_action_with_model(record.state, model,training)

        if type(values_var) == type(None) and action_id == -1:
            return ('fail',-1), None

        if not next_state.verify_game_valid():
            print("Invalid game")

        stat = next_state.game_status()
        rec = Record(record,next_state,values_var,action_id,model,stat,record.state.active_player)

        if np.random.random()>.999 and type(values_var)!=type(None):
            #print('game_state',next_state.arr)
            print('weights norm',sum(torch.norm(p) for p in model.parameters()).data.numpy()[0])
            print('action values',values_var.data.numpy()[:,0])

        if stat[0] == 'won':
            return stat, rec
        elif stat[0] == 'draw':
            return stat, rec

        return stat,rec


    ' return <next_state, values_var, act_id> '
    def sample_action_with_model(self, gstate, model, training=True):
        # Gather future 1-step states
        future_games = gstate.simulate_all_actions(gstate.active_player)
        if all( (fg == None for fg in future_games) ):
            print("all bad")
            return gstate, None, -1 # return old state

        # TODO only compute what works, need to un-hardcode 28x batching in the model code

        gsts = [fg.get_game_features() if fg != None else np.zeros(62) for fg in future_games]
        csts = [fg.get_card_features_as_2d() if fg != None else np.zeros([12,13]) for fg in future_games]
        fsts,gsts = np.array(gsts), np.array(csts)

        # Evaluate future states with the Value Function.
        values = model(fsts, gsts)

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
                record = final_record

                ' These all run in reverse, we are following the linked list backward '
                models = set()
                value_vars = []
                selected_acts = []
                weights = []
                pids = []

                while record != None and record.prev:
                    assert( record.prev != record ) # loop?!

                    if type(record.values_var) != type(None):
                        value_vars.append(record.values_var)
                        selected_acts.append(record.act_id)
                        weights.append(1.0 if record.state.active_player == winner else -.33)
                        pids.append(record.pid)
                        if record.model not in models:
                            models.add(record.model)
                            record.model.zero_grad()
                        record = record.prev

                '''
                2 approaches for loss:
                  1. Some distance loss to wanted value function
                  2. Softmax + xent
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
            for model in models:
                print("stepping with loss:",loss.data[0])
                model.opt.step()

            self.final_states_not_applied = []

            return True
