import numpy as np
from numpy import random as npr
import random,sys
import itertools
from collections import namedtuple
import torch

from game import Actions
from models import PolicyGradientModel
import samplers
from players import BasePlayer

from util import COIN_PICKUPS, one_hot, one_neg, LEVEL_PROBABILITIES
from game import Action

import torch
import torch.autograd as ag

'''
Value Iteration Player.
 - We try to model the value of all states, of course we need to approximate
   the state (in this case with a neural net) b/c there are too many states+noise.
 - To train, we play games and record actions. If we win, we apply the `backup` operator
   and interpolate a high-valued endstate back through all steps. If we lose, we do the
   same with a low-valued endstate.


    find_action :: State -> Action
but first, we approximate with ValueModel: 
    ValueModel :: State -> [(State, Value)] 
and sample a state w/ value as weightings

Note: we don't model actions, but we need to simulate all one-step game states in order
      to compute values (by running the state through ValueModel)
'''

class Player(BasePlayer.BasePlayer):
    def __init__(self,idx,game,use_model=None):
        super().__init__(idx,game,use_model)

    ' important! '
    def reset(self):
        super().reset()
        ' We will also collect bad moves and disencourage them '
        self.bad_move_history = []

    def find_action_with_model(self, game):
        model = self.model

        #ok_cards = [kv for kv in game.cards.items()]
        #ok_cards_ids = [kv[0] for kv in ok_cards]
        #ok_cards_feats = [kv[1].to_features() for kv in ok_cards]

        num_acts = 12+1+15
        vals = []

        '''
        TODO
        encode some logic so that we penalize selecting a bad action, BUT only if
        there was actually a good one we could have chosen.
        I could have another list stored per-game with such choices and down them in apply_reward.
        '''

        '''
        We need to batch all future-states per-player. It is **much** faster.
        Do it by a loop, to simulate all future states, then applying them
        as a batch to the model.
        `vals` will be a list of variables, we keep track of the sampled one (or argmax if none work).
        '''

        f_gsts,f_csts = [],[]
        for act_id in range(num_acts):
            if game.action_is_valid_for_game(self.pid, Actions[act_id]):
                # Estimate value of state
                future_gst,future_csts = game.simulate_action_for_game(self.pid,Actions[act_id])
                #v = self.model(future_gst,future_csts).data.numpy()[0]
                f_gsts.append(future_gst)
                f_csts.append(future_csts)
                #v = self.model(future_gst,future_csts)
                vals.append(None)
            else:
                # TODO: I need to set state to have a lower value.
                # Invalid state, value is 0
                #v = ag.Variable(torch.FloatTensor(0.0),requires_grad=False)
                #v = ag.Variable(torch.FloatTensor([0.0]),requires_grad=True)
                #vals.append(v)
                future_gst,future_csts = game.simulate_action_for_game(self.pid,Action('noop',0,0))
                f_gsts.append(future_gst)
                f_csts.append(future_csts)
                vals.append(None)

        vs = self.model(f_gsts,f_csts)
        _j = 0
        for i in range(len(vals)):
            if vals[i] is None:
                vals[i] = vs[_j]
                _j += 1
            else:
                vals[i] = 1.0

        probs = np.concatenate([v.data.numpy() for v in vals])
        vals = torch.cat(vals)

        probs = probs/sum(probs)

        if sum(probs) < .00000001:
            print("Bad vals")
            #self.history.append( (vals,act_id,trials) )
            #return Action('noop',0,0)



        #vals = torch.cat(vals)
        act,act_id,trials = samplers.proportional_sample(probs,game=game,pid=self.pid)

        if act == 'noop':
            self.history.append( (vals[probs.argmax()],act_id,trials) )
            return Action('noop',0,0)

        ' Only store the chosen action'
        self.history.append( (vals[act_id],act_id,trials ) )

        ' Occasionaly log action distribution '
        self.maybe_log(vals.data.numpy(),game)

        return act


    '''

    Example update:

     time    :  t0  -   t1   -   t2   -  t3  -  t4
     cur val :  .2      .4       .2     .2      .3
     target  :  .2      .3       .4     .5      1.0 
    ------------------------------------------------
     grad    :   0     -.1       +.2    +.3    +.7


    I should experiment with different `target` values:
       - interpolating old/new with discounted reward
       - interpolating old/new, with new coming from future state, recursively
       - Could do a flat +/-
    '''

    def apply_reward(self, did_win, do_step=True):
        model = self.model

        ' Batched ' 
        scores,acts,weights = [],[],[]
        model.zero_grad()
        

        scores = [ev[0] for ev in self.history]
        acts = [int(ev[1]) for ev in self.history]

        ''' Using interpolated-from-future values (TESTED: converges to uniform dist) '''
        '''
        last_reward = 10.0 if did_win else 0.1
        targets = [last_reward]

        alpha,beta = .6,.4 # Interpolation weights, higher alpha => faster changing

        for t,ev in enumerate(self.history[-2::-1]): # going backwards, skipping last
            #targets.append( scores[t][acts[t]].data[0]*beta + last_reward*alpha )
            targets.append( scores[t].data[0]*beta + last_reward*alpha )
            last_reward = targets[-1]
        targets = targets[::-1]
        '''

        ''' Using interpolated old/new wrt. nearly-constant new'''
        targets = np.linspace(8 ,10, len(self.history)) if did_win \
            else  np.linspace(.5,.0001, len(self.history)) # losing doesn't mean initial choices were bad

        #active_scores = torch.cat( [sc[act_id] for (sc,act_id) in zip(scores,acts)] )
        active_scores = torch.cat(scores)

        print("OLDVALS",active_scores.data.numpy())
        #print('TARGETS',targets)

        targets = ag.Variable(torch.FloatTensor(targets))


        loss = torch.nn.functional.mse_loss(active_scores,targets)

        if do_step:
            loss.backward()
            model.opt.step()
        else:
            return loss
