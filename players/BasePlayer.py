import numpy as np
from numpy import random as npr
import random,sys
import itertools
from collections import namedtuple
import torch
from util import one_hot

from game import Actions



class BasePlayer(object):
    '''
    :use_model is either a type or an instance of from /models/*
    '''
    def __init__(self,idx,game,use_model=None):
        self.game = game
        self.reset()
        self.pid = idx

        if type(use_model) == type:
            self.model = use_model()
        else:
            self.model = use_model

    def reset(self):
        self.coins = np.zeros(6) # resource values
        self.cards = np.zeros(5) # resource values
        self.points = 0
        self.reserves = []
        self.history = []
        self._feats_cache = None


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
        self._feats_cache = None
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

    ' return np.arr of [coins,cards,points],dbank as if action was executed'
    def simulate_action_for_player(self,action):
        coins = np.copy(self.coins)
        cards = np.copy(self.cards)
        points = self.points
        dbank = np.zeros(6)
        if action.type == 'card':
            # 1. subtract coins needed to buy it
            card = self.game.cards[action.card_id]
            cost = card.cost_6()
            cost = cost - self.cards_6() # We don't need to pay if we have cards
            cost = np.clip(cost, 0,99) 
            dbank = cost
            coins -= cost
            # if any are negative, use gold
            neg_sum = sum(coins[coins<0])
            for i,c in enumerate(coins):
                if c < 0: dbank[i] += c
            dbank[-1] = -neg_sum
            coins = np.clip(coins,0,99)
            coins[-1] += neg_sum # use gold
            coins = np.clip(coins,0,99)

            # 2. increment points & cards
            points += points
            cards += one_hot(card.resource)
        elif action.type == 'coins':
            # 1. increment coins
            coins += action.coins
            dbank = -action.coins
        elif action.type == 'reserve':
            dbank = np.zeros(6)
            card = self.game.cards[action.card_id]
            #self.reserves.append(card)
            coins[-1] += 1
        return np.concatenate([coins,cards,[points]]),dbank



    def to_features(self):
        if self._feats_cache:
            return self._feats_cache
        return np.concatenate([self.coins,self.cards,[self.points]])
    def cards_6(self):
        return np.concatenate([self.cards,[0]])

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "<Player %d, pts=%d, coins=%s, cards=%s>"%(self.pid,self.points,
                ' '.join(map(str,self.coins.astype(int))), ' '.join(map(str,self.cards.astype(int))))

    def maybe_log(self, scores, game):
        eps = np.random.random()
        if hash(eps) % 50000 == 0:
            print("Scores on turn",game.turn_number)
            for i in range(len(scores)):
                print(' ',Actions[i],"=>",scores[i])
                if game.logf:
                    game.logf.write(' '+str(Actions[i])+"=>"+str(scores[i])+"\n")

