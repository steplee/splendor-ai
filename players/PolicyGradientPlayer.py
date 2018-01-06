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

'''
Direct Policy-Gradient player.
We do not model V or Q functions, just take game-state to a softmax over actions
'''

class Player(BasePlayer.BasePlayer):
    def __init__(self,idx,game,use_model=None):
        super().__init__(idx,game,use_model)

    def find_action_with_model(self, game):
        model = self.model

        # 2. Find which cards we can pickup
        #ok_cards = [kv for kv in game.cards.items() if game.action_is_valid_for_game( self.pid, Action(type='card',card_id=kv[0],coins=-1))]
        ok_cards = [kv for kv in game.cards.items()]
        ok_cards_ids = [kv[0] for kv in ok_cards]
        ok_cards_feats = [kv[1].to_features() for kv in ok_cards]


        # 3. Compute scores
        gst = game.to_features()[0]
        #score_idx = model(gst, ok_cards_feats)
        scores = model(gst, ok_cards_feats)
        nscores = np.copy(scores.data.numpy()[0])

        # 4. Find act
        act,act_id,trials = samplers.proportional_sample(nscores, pid=self.pid, game=game)

        if act == 'noop':
            self.history.append( (scores,act_id,trials) )
            return Action('noop',0,0)


        # 5. Store in history
        self.history.append( (scores,act_id,trials ) )

        ' Occasionaly log action distribution '
        self.maybe_log(scores.data.numpy()[0],game)

        return act

    ' Apply xent loss to each entry in history to raise/lower prob of selection again '
    ' :do_step whether or not to apply the loss step, or just return the loss for future use'
    def apply_reward(self, did_win, do_step=True):
        model = self.model

        ' Batched ' 
        scores,acts,weights = [],[],[]
        model.zero_grad()
        
        # This code is vectorized below
        '''
        for t,ev in enumerate(self.history):
            score_var,act_id,trials = ev
            scores.append(score_var)
            acts.append(int(act_id))
            if act_id == 'noop' or trials > 200:
                weights.append( -.3 ) # decrease the argmax act_id
            else:
                weights.append( 1.0 if did_win else -.33 )
        '''
        scores = [ev[0] for ev in self.history]
        acts = [int(ev[1]) for ev in self.history]
        weights = [(-.3 if ev[1] == 'noop' or ev[2] > 200 else (1.0 if did_win else -.33)) for ev in self.history]


        scores = torch.cat(scores)
        weights = torch.autograd.Variable(torch.FloatTensor(weights),requires_grad=False)

        target = torch.autograd.Variable(torch.LongTensor(acts),requires_grad=False)
        loss = torch.nn.functional.nll_loss(scores, target, size_average=False) * weights
        loss = torch.sum(loss)

        if do_step:
            loss.backward()
            model.opt.step()
        else:
            return loss
