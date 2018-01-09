import numpy as np
from numpy import random as npr
import random,sys,os,time
import itertools,optparse
from collections import namedtuple
import torch

from util import COIN_PICKUPS, one_hot, \
                LEVEL_PROBABILITIES, LEVEL_TEMPLATES, Actions, Action





''' - -  -  -  -  -  -  GAME  -  -  -  -  -  -  -  -  -  -  '''

log_scores = True

class GameState(object):
    def __init__(self, to_copy=None):
        if to_copy:
            self.num_players = to_copy.num_players
            self.active_player = to_copy.active_player
            self.noops_in_row = to_copy.noops_in_row
            self.arr = np.copy(to_copy.arr)
            self.card_arr = np.copy(to_copy.card_arr)
        else:
            self.num_players = 4
            self.active_player = 0
            self.noops_in_row = 0
            self.arr = np.zeros(62) # bank x6, players 52x, active_player x4
            bank = self.get_bank()
            bank[...] = np.ones(6) * 7
            self.get_bank()[-1] = 5
            bank[-1] = 5
            self.init_cards()

    def action_is_valid_for_game(self,pid,action):
        if action.type == 'noop':
            return True

        cost = self.get_card_cost(action.card_id)
        pcoins = self.get_player_coins(pid)

        if action.type == 'card':
            pcards = self.get_player_cards(pid)
            diff = pcoins+pcards - cost
            return (all(diff >= 0) or -sum(diff[diff<0]) <= pcoins[-1]) \
               and True # TODO allow a limited set of cards
        elif action.type == 'coins':
            return (sum(pcoins+action.coins) <= 10) \
               and all(self.get_bank()-action.coins >= 0) \
               and (all(action.coins <= 1) or any(self.get_bank()*action.coins>4))
        elif action.type == 'reserve':
            return sum(pcoins+action.coins) <= 10 \
               and pcoins[-1] < 3 \
               and self.get_bank()[-1] >= 1 and True # TODO: check reserved card is not empty

    def execute_action_for_game(self, pid, action):
        if action.type == 'noop':
            self.noops_in_row += 1
        else:
            self.noops_in_row = 0

        if not self.action_is_valid_for_game(pid,action):
            return None

        if action.type == 'card':
            bank = self.get_bank()

            # 1. Player
            card_cost = self.get_card_cost(action.card_id)
            card_cost = card_cost - self.get_player_cards(pid) # actually deep copy
            card_cost = np.clip(card_cost,0,99)
            player_coins = self.get_player_coins(pid)
            player_coins[...] = player_coins - card_cost

            neg_sum = np.sum(player_coins[player_coins<0])
            for i,c in enumerate(player_coins):
                if c<0: bank[i] += c

            player_coins[...] = np.clip(player_coins,0,99)
            player_coins[-1] += neg_sum # use gold
            player_coins[...] = np.clip(player_coins,0,99)

            self.increase_player_points(pid, self.get_card_points(action.card_id))
            self.get_player_cards(pid)[...] += self.get_card_resource(action.card_id)

            # 2. Game
            bank += card_cost
            bank[-1] += -neg_sum
            self.get_card_whole(action.card_id)[...] = self.next_card(action.card_id//4)
        elif action.type == 'coins':
            # 1. Player
            self.get_player_coins(pid)[...] += action.coins
            # 2. Game
            self.get_bank()[...] -= action.coins
        elif action.type == 'reserve':
            # 1. Player
            self.get_player_coins(pid)[-1] += 1
            self.get_bank()[-1] -= 1
        self.active_player = (self.active_player + 1) % 4
        
        return self

    def simulate_all_actions(self, pid):
        # TODO a micro-optizmization is in order: no need to copy if action invalid.
        return [GameState(to_copy=self).execute_action_for_game(pid,act) for act in Actions]

    def init_cards(self):
        self.card_arr = np.zeros((12*13))
        for lvl in range(3):
            for i in range(4):
                self.get_card_whole(lvl*4+i)[...] = self.next_card(lvl)


    def next_card(self, lvl):
        # TODO: I need to have a queue of cards to draw from.
        # When we simulate, we must draw the actual next card.

        template_id = np.random.choice(range(len(LEVEL_TEMPLATES[lvl])),p=LEVEL_PROBABILITIES[lvl])
        template = LEVEL_TEMPLATES[lvl][template_id]
        assign = np.random.choice(range(5),len(template[1]))
        cost = np.zeros(6)
        pts = template[2]

        for i,j in enumerate(assign):
            cost[j] = template[1][i]
        resource = np.random.randint(5)
        #card = Card(template[2], resource, cost)

        return np.concatenate([one_hot(resource,size=6),cost,[pts]])

    def game_status(self):
        winners = [pid for pid in range(self.num_players) if self.get_player_points(pid)>=15]
        if len(winners) > 0:
            return ('won', winners[0])
        elif self.noops_in_row > 4:
            return ('draw', -1)
        else:
            return ('ongoing',-1)


    ''' Getters / Setters '''

    def get_bank(self): return self.arr[:6]
    def get_active_player(self): return self.arr[-5:]

    def get_card_whole(self, c): return self.card_arr[c*13:(c+1)*13]
    def get_card_points(self, c): return self.card_arr[c*13+12]
    def get_card_resource(self,c): return self.card_arr[c*13:c*13+6]
    def get_card_cost(self,c): return self.card_arr[c*13+6:c*13+12]

    # Player: <coins x6, cards x6, points>
    def get_player_coins(self,p): return self.arr[6+p*13:6+p*13+6]
    def get_player_cards(self,p): return self.arr[6+p*13+6:6+p*13+12]
    def get_player_points(self,p): return self.arr[6+p*13+12]
    def increase_player_points(self,p,delta): self.arr[6+p*13+12] += 1

    def get_game_features(self): return self.arr
    def get_card_features_as_2d(self): return np.reshape(self.card_arr,[12,13])



    ''' Debugging '''

    ' Make sure all coins are in circulation still '
    def verify_game_valid(self):
        coins = np.ones(6)*7
        coins[-1] = 5
        
        for p in range(self.num_players):
            coins -= self.get_player_coins(p)
        coins -= self.get_bank()

        if all(coins == 0):
            return True
        else:
            print("ERROR, coins are foobarred, difference: ",str(coins))
            sys.exit(1)
            return False
