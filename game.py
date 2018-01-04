import numpy as np
from numpy import random as npr
import random,sys,os,time
import itertools,optparse
from collections import namedtuple
import torch

import model

from util import COIN_PICKUPS, one_hot, \
                LEVEL_PROBABILITIES, LEVEL_TEMPLATES

from policies import pg_player as player


''' - -  -  -  -  -  -  CARD  -  -  -  -  -  -  -  -  -  -  '''

#                           int        index      array
#Card = namedtuple('Card',['points', 'resource', 'cost'])
class Card():
    def __init__(self, pts,res,cost):
        self.points = pts
        self.resource = res
        self.cost = cost
    def to_features(self):
        return np.concatenate([ [self.points], one_hot(self.resource,6), self.cost])
    def cost_6(self):
        a = np.zeros(6)
        a[0:5] = self.cost
        return a

''' - -  -  -  -  -  -  ACTION  -  -  -  -  -  -  -  -  -  -  '''

#                               str      int       array
Action = namedtuple('Action',['type', 'card_id', 'coins'])

'''
The totality of actions is:
    10 (pickup 5 choose 3 coins)
    5  (pickup 2 of one coin)
    12 (pickup a card)
    5  (reserve a card)
    1  (swap randomly when full 10 coins)


 =  33 TOTAL (maximum, depending on valid cards)
So I need a 33-dim softmax output from my model.
Each component will be normalized prob of selecting action i.
Order is same as above (10-5-12-1)
'''

def sample_random_action():
    if np.random.randint(10) < 7: # coins
        pickup = random.choice(COIN_PICKUPS)
        return Action('coins', card_id=-1, coins=pickup)
    else:
        return Action('card', card_id=(np.random.randint(3),np.random.randint(3)), coins=-1)




''' - -  -  -  -  -  -  GAME  -  -  -  -  -  -  -  -  -  -  '''

log_scores = True

optp = optparse.OptionParser()
optp.add_option('--name', default=str(time.time()).split('.')[0])
opts = optp.parse_args(sys.argv)[0]

class Game(object):
    def __init__(self):
        self.num_players = 4
        self.active_player = 0

        ' if :use_model is None, each player will create its own, or we can share by having game.model '
        self.model = model.Net()
        self.model.train()
        self.players = [player.Player(i,self,use_model=self.model) for i in range(self.num_players)]
        #self.players = [player.Player(i,self,use_model=None) for i in range(self.num_players)]

        self.bank = np.ones(6)*7
        self.bank[-1] = 5 # gold
        self.cards = {}
        self.init_cards()
        self.turn_number = 0

        #self.json_history = [] # for fallback
        #self.step_history = [] # for training
        #self.entropy_log = []

    ''' Game Logic '''

    def action_is_valid_for_game(self,player_turn,action):
        # 1. be valid for player
        # if buying a card:
            # 2. card id exists
        # if picking from bank:
            # 2. bank has the coins
            # 3. bank has >4 if it is a 2 pickup
            # 4. we do not pick up any gold
        # if reserve:
            # 2. card exists
            # 3. at least one gold chip
        if action.type == 'noop':
            return True

        if not self.players[player_turn].action_is_valid_for_player(action):
            return False

        if action.type == 'card':
            return self.cards[action.card_id] != None
        elif action.type == 'coins':
            return all(self.bank-action.coins >= 0) and \
                   all(action.coins <= 2) and \
                   sum(action.coins) <= 3 and \
                   (len(action.coins) < 6 or action.coins[-1] == 0) and \
                  (all(action.coins <= 1) or any(self.bank*action.coins>4)) # 2 left rule
        elif action.type == 'reserve':
            return self.bank[-1] >= 1 and self.cards[action.card_id] != None

    def execute_action_for_game(self,player_turn,action,log=True):
        if action.type == 'noop':
            self.noops_in_row += 1
            if self.noops_in_row > 10: # error in game, throw it away!
                return False
        else:
            self.noops_in_row = 0
        # 1. execute for player, get dbank & apply
        dbank = self.players[player_turn].execute_action_for_player(action)
        self.bank += dbank
        if action.type == 'card':
            # 2. remove card and replace it
            self.cards[action.card_id] = self.next_card(action.card_id[0])
        elif action.type == 'coins':
            pass # do nothing, dbank is all
        elif action.type == 'reserve':
            # 2. replace card
            self.cards[action.card_id] = self.next_card(action.card_id[0])
            # 3. decrement gold by one
            self.bank[-1] -= 1
        if log:
            print("Executing action " + str(action))
            print("             ->  " + str(self.players[player_turn]))
        return True

    def next_card(self, lvl):
        template_id = np.random.choice(range(len(LEVEL_TEMPLATES[lvl])),p=LEVEL_PROBABILITIES[lvl])
        template = LEVEL_TEMPLATES[lvl][template_id]
        assign = np.random.choice(range(5),len(template[1]))
        cost = np.zeros(5)
        for i,j in enumerate(assign):
            cost[j] = template[1][i]
        resource = np.random.randint(5)
        card = Card(template[2], resource, cost)
        return card

    def init_cards(self):
        for lvl in range(3):
            for i in range(4):
                self.cards[lvl,i] = self.next_card(lvl)

    ' Make sure all coins are in circulation still '
    def verify_game_valid(self):
        coins = np.ones(6)*7
        coins[-1] = 5
        
        for p in self.players:
            coins -= p.coins
        coins -= self.bank

        if all(coins == 0):
            return True
        else:
            print("ERROR, coins are foobarred, difference: ",str(coins))
            sys.exit(1)
            return False


    ''' Serialization'''

    def to_json(self):
        pass # TODO
    def from_json(self,jobj):
        pass

    def reset(self):
        self.num_players = 4
        self.active_player = 0
        for p in self.players:
            p.reset()
        self.bank = np.ones(6)*7
        self.bank[-1] = 5 # gold
        self.cards = {}
        self.init_cards()


    ''' Running the game '''


    def game_status(self):
        winners = [pid for (pid,p) in enumerate(self.players) if p.points>=30]
        if len(winners) > 0:
            return ('won', winners[0])
        else:
            return ('in-progress',)


    # Play many games using `play_entire_game()`, reusing the same game py object
    def play_many_games(self, games=500):
        with open(os.path.join('logs',opts.name),'w') as logf:
            self.logf = logf
            for game in range(games):
                ret = self.play_entire_game(log=False)
                if ret != 'draw':
                    winner,turns = ret
                    #norm2 = sum((torch.norm(p) for p in self.model.parameters())).data[0]
                    norm2 = sum(sum((torch.norm(p) for p in pl.model.parameters())).data[0] for pl in self.players)
                    print("game ",game,"(",turns," turns) norm after ",norm2)
                    logf.write("game "+str(game)+"( "+str(turns)+" turns) norm after "+str(norm2)+"\n")
                self.reset()

                if game > 5 and game % 100 == 0: # Eval.
                    avg = 0
                    cnt = 10
                    for egame in range(10):
                        ret = self.play_entire_game(log=False)
                        if ret != 'draw':
                            _,turns = ret
                            avg += turns
                        else:
                            cnt -= 1
                        self.reset()
                    print("At game "+str(game)+" average turns: {"+str( (avg/float(cnt)))+"}")
                    logf.write("At game "+str(game)+" average turns: {"+str( (avg/float(cnt)))+"}\n")
                    self.model.train(True)
                if game > 5 and game % 200 == 0: # Save model
                    torch.save(self.model, os.path.join('saved_models',opts.name))
            self.logf = None


    def play_entire_game(self,log=True):
        for turn in range(200):
            self.turn_number = turn
            if log:
                print("\nTURN",turn,)
                print(str(self),"\n")
            for pid in range(4):
                self.active_player = pid
                act = self.players[pid].find_action_with_model(self)

                if not self.execute_action_for_game(pid, act,log=log):
                    return 'draw'
                # TODO remember this is disabled
                #self.verify_game_valid()

                status = self.game_status()
                if status[0] == 'won':
                    print('Player %d won on turn %d !!!!!'%(status[1],turn))

                    self.players[status[1]].apply_reward(did_win=True)
                    '''
                    for i in range(self.num_players):
                        if status[1] != i:
                            self.players[i].apply_reward(did_win=False)
                    '''


                    return status[1], turn
        return 'draw'


    ' Return a 2-tuple of <gst,cst>, see model.py for the definitions '
    def to_features(self, cards=None):
        if cards is None:
            cards = self.cards.values()
        cards = np.concatenate([card.to_features() for card in cards])
        players = np.concatenate([player.to_features() for player in self.players])
        active_player = one_hot(self.active_player, 4)
        return np.concatenate([self.bank, players, active_player]), cards

    def __str__(self):
        return "Game bank=%s, cards=%d"%(str(self.bank), len([it for it in self.cards.items() if it[1] != None]))



''' - -  -  -  -  -  -  DRIVER  -  -  -  -  -  -  -  -  -  -  '''

def run_many_games():
    g = Game()
    g.play_many_games(50000)

if __name__=='__main__' and 'train' in sys.argv:
    run_many_games()
