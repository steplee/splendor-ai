import numpy as np
from numpy import random as npr
import random,sys
import itertools
from collections import namedtuple
import torch

import model

from util import COIN_PICKUPS, one_hot, one_neg, LEVEL_PROBABILITIES
from game import Action

'''
Direct Policy-Gradient player.
We do not model V or Q functions, just take game-state to a softmax over actions
'''

class Player(object):
    def __init__(self,idx,game,use_model=None):
        self.game = game
        self.reset()
        self.pid = idx
        self.model = use_model if use_model is not None else model.Net() 

    def reset(self):
        self.coins = np.zeros(6) # resource values
        self.cards = np.zeros(5) # resource values
        self.points = 0
        self.reserves = []
        self.history = []


    def action_is_valid_for_player(self,action):
        # if buying a card:
            # 1. we have enough coins
        # if picking from bank:
            # 1. we have < 10
            # 2. we pick 3 or 2
        # if reserving:
            # 1. we have < 3 reserves
            # 2. we have < 10 coins
        gold = self.coins[-1]
        if action.type == 'card':
            cost = self.game.cards[action.card_id].cost_6()
            diff = self.coins[0:5]+self.cards - cost[0:5]
            return all(diff >= 0) or -sum(diff[diff<0]) <= self.coins[-1] # or gold can fill in
            #return all([ (self.coins+self.cards)[i] >= cost for i in range(5)])
        elif action.type == 'coins':
            good = sum(self.coins+action.coins) <= 10
            good &= sum(action.coins) <= 3
            good &= all( (x <= 2) for x in action.coins)
            return good
        elif action.type == 'reserve':
            good = sum(self.coins+action.coins) <= 10
            good &= len(self.reserves) < 3
            return good

    # Returns any change that should be made to game's bank
    def execute_action_for_player(self,action):
        dbank = np.zeros(6)
        if action.type == 'card':
            # 1. subtract coins needed to buy it
            card = self.game.cards[action.card_id]
            cost = card.cost_6()
            cost = cost - self.cards_6() # We don't need to pay if we have cards
            cost = np.clip(cost, 0,99) 
            dbank = cost
            self.coins -= cost
            # if any are negative, use gold
            neg_sum = sum(self.coins[self.coins<0])
            for i,c in enumerate(self.coins):
                if c < 0: dbank[i] += c
            dbank[-1] = -neg_sum
            #print("ns",neg_sum,"for card",str(card.cost),"with coins",str(self.coins))
            self.coins = np.clip(self.coins,0,99)
            self.coins[-1] += neg_sum # use gold
            #print("ns",neg_sum,"for card",str(card.cost),"with coins",str(self.coins))
            self.coins = np.clip(self.coins,0,99)

            # 2. increment points & cards
            self.points += card.points
            self.cards += one_hot(card.resource)
        elif action.type == 'coins':
            # 1. increment coins
            self.coins += action.coins
            dbank = -action.coins
        elif action.type == 'reserve':
            dbank = np.zeros(6)
            card = self.game.cards[action.card_id]
            self.reserves.append(card)
            self.coins[-1] += 1
        return dbank

    def find_action_with_model(self, game):
        model = self.model

        # 2. Find which cards we can pickup
        #ok_cards = [kv for kv in game.cards.items() if game.action_is_valid_for_game( self.pid, Action(type='card',card_id=kv[0],coins=-1))]
        ok_cards = [kv for kv in game.cards.items()]
        ok_cards_ids = [kv[0] for kv in ok_cards]
        ok_cards_feats = [kv[1].to_features() for kv in ok_cards]

        def decode_act_id(act_id):
            if act_id < 10:
                act = Action('coins', card_id=-1, coins=(COIN_PICKUPS[act_id]))
            elif act_id < 15:
                act = Action('coins', card_id=-1, coins=one_hot(act_id-10,6)*2)
            elif act_id < 16:
                act = Action('reserve', card_id=(np.random.randint(3),np.random.randint(3)),coins=-1)
            else:
                card_id = ok_cards_ids[act_id-16]
                act = Action('card', card_id=card_id, coins=-1)
            #print(str(act),act_id)
            return act

        # 3. Compute scores
        gst = game.to_features()[0]
        #score_idx = model(gst, ok_cards_feats)
        scores = model(gst, ok_cards_feats)
        nscores = np.copy(scores.data.numpy()[0])

        # 4. Find act
        ' According to our policy type, we could choose highest-scoring (test) '
        ' or sample from the softmax (train), or a more elaborate scheme, like epsilon-greedy'
        '''
        taken_act = None
        taken_act_id = None
        for (scores,act_id) in score_idx:
            act = decode_act_id(act_id)
            print("trying %d -> %s"%(act_id,str(act)))
            if game.action_is_valid_for_game(self.pid, act):
                taken_act = act
                taken_act_id = act_id
                break

        if not taken_act:
            print("Game", str(game))
            raise Exception("Failed to find good action")
        '''

        ''' WEIGHTED-SAMPLE '''
        #'''
        trials = 0
        act,act_id = None,None
        while act is None or not game.action_is_valid_for_game(self.pid,act):
            if trials <= 100:
                act_id = int(np.random.choice(range(len(nscores)),1,p=nscores)[0])
            else:
                act_id = int(np.random.choice(range(len(nscores)),1)[0])
            #print("test",act_id)
            act = decode_act_id(act_id)
            trials += 1
            if trials >= 300:
                print(nscores)
                print(" BAD: performing a noop!")
                self.history.append('noop')
                return Action('noop',0,0)
        #'''

        eps = np.random.random()
        ''' EPSILON-GREEDY '''
        '''
        act,act_id = None,None
        #if eps > .76:
        if eps > .99: # TODO start low, get bigger
            # greedy action, choose best that is valid
            while act is None or not game.action_is_valid_for_game(self.pid,act):
                act_id = int(np.argmax(nscores))
                if nscores[act_id] == -1:
                    break
                nscores[act_id] = -1
                #print("test",act_id)
                act = decode_act_id(act_id)
        if act is None or eps <= .100001:
            # choose random
            trials = 0
            act,act_id = None,None
            while act is None or not game.action_is_valid_for_game(self.pid,act):
                act_id = int(np.random.choice(range(len(nscores)),1)[0])
                act = decode_act_id(act_id)
                trials += 1
                if trials >= 200:
                    print(nscores)
                    print(" BAD: performing a noop!")
                    self.history.append('noop')
                    return Action('noop',0,0)
        '''

        # 5. Store in history
        self.history.append( (scores,act_id) )
        ' Occasionaly log action distribution '
        if hash(eps) % 100000 == 0:
            nnscore = scores.data.numpy()
            print("Scores on turn",game.turn_number)
            for i in range(len(nscores)):
                print(' ',decode_act_id(i),"=>",nnscore[0][i])
                if game.logf:
                    game.logf.write(' '+str(decode_act_id(i))+"=>"+str(nnscore[0][i])+"\n")

        return act

    ' Apply xent loss to each entry in history to raise/lower prob of selection again '
    # xent to dis/encourage same action being chosen again
    def apply_reward(self, did_win):
        model = self.model

        ' Batched ' 
        scores,acts = [],[]
        model.zero_grad()
        for t,ev in enumerate(self.history):
            if ev == 'noop':
                pass
            else:
                score_var,act_id = ev
                scores.append(score_var)
                acts.append(act_id)

        scores = torch.cat(scores)

        if did_win:
            target = torch.autograd.Variable(torch.LongTensor(acts),requires_grad=False)
            #loss = torch.nn.functional.cross_entropy(scores, target) * 3.0
            loss = torch.nn.functional.nll_loss(scores, target) * 3.0
        else:
            target = torch.autograd.Variable(torch.LongTensor(acts),requires_grad=False)
            #loss = -torch.nn.functional.cross_entropy(scores, target)
            loss = -torch.nn.functional.nll_loss(scores, target) * .2

        loss.backward()
        model.opt.step()

        ' Not Batched '
        """
        for t,ev in enumerate(self.history):
            if ev == 'noop':
                pass
            else:
                score_var,act_id = ev

                if did_win:
                    #norm1 = sum((torch.norm(p) for p in model.parameters())).data[0]
                    model.zero_grad()
                    #print("Encouraging",act_id)

                    # Hinge loss
                    '''
                    target = one_hot(act_id, size=score_var.size()[1]).astype(int)
                    target = torch.autograd.Variable(torch.FloatTensor(target),requires_grad=False)
                    target = target.resize(1,score_var.size()[1])
                    loss = torch.nn.functional.hinge_embedding_loss(score_var, target) * 2
                    #target = torch.autograd.Variable(torch.LongTensor([act_id]),requires_grad=False)
                    #loss = torch.nn.functional.multi_margin_loss(score_var, target) * 2
                    '''


                    # XEnt loss
                    #'''
                    target = torch.autograd.Variable(torch.LongTensor([act_id]),requires_grad=False)
                    loss = torch.nn.functional.cross_entropy(score_var, target)
                    #'''

                    loss.backward()
                    model.opt.step()
                else:
                    model.zero_grad()

                    # XEnt loss
                    target = torch.autograd.Variable(torch.LongTensor([act_id]),requires_grad=False)
                    loss = - torch.nn.functional.cross_entropy(score_var, target)

                    #'''
                    ' A strong regularizer may help us avoid mode collapse to one action '
                    #loss = sum(torch.nn.functional.mse_loss(p,torch.autograd.Variable(torch.zeros(p.size()))) for p in model.parameters()) * .5
                    #loss.backward()
                    #model.opt.step()

                    ' Or apply hinge loss to select anything, averaging out bad action '
                    # Hinge loss
                    '''
                    #target = np.ones(score_var.size()[1], dtype=int) / score_var.size()[1]
                    target = np.ones(score_var.size()[1], dtype=int)
                    target = torch.autograd.Variable(torch.LongTensor(target),requires_grad=False)
                    target = target.resize(1,score_var.size()[1])
                    loss = torch.nn.functional.multilabel_margin_loss(score_var, target) * .1
                    '''

                    loss.backward()
                    model.opt.step()
        """



            


    def to_features(self):
        return np.concatenate([self.coins,self.cards,[self.points]])
    def cards_6(self):
        return np.concatenate([self.cards,[0]])

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "<Player %d, pts=%d, coins=%s, cards=%s>"%(self.pid,self.points,
                ' '.join(map(str,self.coins.astype(int))), ' '.join(map(str,self.cards.astype(int))))
