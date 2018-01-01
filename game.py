import numpy as np
from numpy import random as npr
import random,sys
import itertools
from collections import namedtuple
import torch

import model

i32 = np.int32

def one_hot(i,size=5):
    a = np.zeros(size)
    if type(i) == int:
        a[i] = 1
    else:
        for j in i:
            a[int(j)] = 1
    return a

COLORS = list('RGWBK')
COLOR_TO_IDX = dict(zip(range(100),COLORS))
IDX_TO_COLOR = dict(zip(COLORS,range(100)))
COIN_PICKUP_CHOICES = list(itertools.combinations(COLORS,3))
COIN_PICKUPS = (itertools.combinations(range(5),3))
COIN_PICKUPS = list(map(one_hot, COIN_PICKUPS))
COIN_PICKUPS = [np.concatenate([x,[0]]) for x in COIN_PICKUPS]

LEVEL_NUM_CARDS = [43, 20, 15]
# Each level is list of tuple <weight, cost, points>
# Since colors are exchangeable, the `cost` is color independent
LEVEL_TEMPLATES = [
        [ (1,[1,1,1,1],0), (1,[1,1,1],0), (.2,[2,2,1],1) ],
        [ (1,[2,2,3],2), (1,[3,3,2],2), (.2,[6],3) ],
        [ (1,[7],4) ]
]

LEVEL_PROBABILITIES = [[],[],[]]
# Normalize
for lvl in range(3):
    row_sum = sum([t[0] for t in LEVEL_TEMPLATES[lvl]])
    for j,t in enumerate(LEVEL_TEMPLATES[lvl]):
        LEVEL_PROBABILITIES[lvl].append( t[0]/row_sum )


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
#                               str      int       array
Action = namedtuple('Action',['type', 'card_id', 'coins'])

def sample_random_action():
    if np.random.randint(10) < 7: # coins
        pickup = random.choice(COIN_PICKUPS)
        return Action('coins', card_id=-1, coins=pickup)
    else:
        return Action('card', card_id=(np.random.randint(3),np.random.randint(3)), coins=-1)


'''
The totality of actions is:
    10 (pickup 5 choose 3 coins)
    5  (pickup 2 of one coin)
    12 (pickup a card)
    5  (reserve a card)
    1  (swap randomly when full 10 coins)


 =  33 TOTAL
So I need a 33-dim softmax output from my model.
Each component will be normalized prob of selecting action i.
Order is same as above (10-5-12-1)
'''



class Player(object):
    def __init__(self,idx,game):
        self.game = game
        self.pid = idx
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
            return all(diff >= 0) or sum(diff[diff<0])+self.coins[-1] >= 0 # or gold can fill in
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
            self.coins = np.clip(self.coins,0,99)
            self.coins[-1] += neg_sum # use gold
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

    def find_action_with_model(self, game, model):

        # 2. Find which cards we can pickup
        ok_cards_ids = [kv[0] for kv in game.cards.items() if game.action_is_valid_for_game(
                    self.pid, Action(type='card',card_id=kv[0],coins=-1))]
        ok_cards_feats = [kv[1].to_features() for kv in game.cards.items() if game.action_is_valid_for_game(
                    self.pid, Action(type='card',card_id=kv[0],coins=-1))]

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
            return act

        # 3. Compute scores
        gst = game.to_features()[0]
        #score_idx = model(gst, ok_cards_feats)
        scores = model(gst, ok_cards_feats)
        nscores = scores.data.numpy()[0]

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

        ''' SAMPLE '''
        trials = 0
        act,act_id = None,None
        while act is None or not game.action_is_valid_for_game(self.pid,act):
            act_id = int(np.random.choice(range(len(nscores)),1,p=nscores)[0])
            #print("test",act_id)
            act = decode_act_id(act_id)
            trials += 1
            if trials >= 300:
                print(nscores)
                print(" BAD: performing a noop!")
                self.history.append('noop')
                return Action('noop',0,0)

        ''' EPSILON-GREEDY '''
        # TODO
        '''
        trials = 0
        eps = np.random.random()
        act,act_id = None,None
        if eps < .1:
            act_id = 123
        '''

        # 5. Store in history
        #self.history.append( (score_idx,act_id) )
        self.history.append( (scores,act_id) )

        return act

    ' Apply xent loss to each entry in history to raise/lower prob of selection again '
    ' We modify based on the action we chose '
    # xent to dis/encourage same action being chosen again
    def apply_reward(self, model, did_win):
        for t,ev in enumerate(self.history):
            if ev == 'noop':
                pass
            else:
                score_var,act_id = ev

                if did_win:
                    #norm1 = sum((torch.norm(p) for p in model.parameters())).data[0]
                    model.zero_grad()
                    #print("Encouraging",act_id)
                    target = torch.autograd.Variable(torch.LongTensor([act_id]),requires_grad=False)
                    loss = torch.nn.functional.cross_entropy(score_var, target)
                    loss.backward() # TODO: do I need retain graph?
                    model.opt.step()
                    #norm2 = sum((torch.norm(p) for p in model.parameters())).data[0]
                    #print("Optim time ",t," before norm ", norm1, " after ",norm2)
                else:
                    model.zero_grad()
                    ' A strong regularizer may help us avoid mode collapse to one action '
                    loss = sum(torch.nn.functional.mse_loss(p,torch.autograd.Variable(torch.zeros(p.size()))) for p in model.parameters()) * 2
                    loss.backward()
                    model.opt.step()
                    # target = np.ones(len(score_var)) / len(score_var)



            


    def to_features(self):
        return np.concatenate([self.coins,self.cards,[self.points]])
    def cards_6(self):
        return np.concatenate([self.cards,[0]])

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "<Player %d, pts=%d, coins=%s, cards=%s>"%(self.index,self.points,
                ' '.join(map(str,self.coins.astype(int))), ' '.join(map(str,self.cards.astype(int))))



class Game(object):
    def __init__(self):
        self.num_players = 4
        self.active_player = 0
        self.players = [Player(i,self) for i in range(self.num_players)]
        self.bank = np.ones(6)*7
        self.bank[-1] = 5 # gold
        self.cards = {}
        self.init_cards()

        self.model = model.Net()
        #self.net = SplendorModel()

        self.json_history = [] # for fallback
        self.step_history = [] # for training
        self.entropy_log = []

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
        if log: print("Executing action " + str(action))
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


    ''' Serialization'''

    def to_json(self):
        pass
    def from_json(self,jobj):
        pass

    def reset(self):
        self.num_players = 4
        self.active_player = 0
        self.players = [Player(i,self) for i in range(self.num_players)]
        self.bank = np.ones(6)*7
        self.bank[-1] = 5 # gold
        self.cards = {}
        self.init_cards()


    ''' AI '''

    def play_entire_game_random(self):
        turn = 0
        for turn in range(150):
            print("\nTurn %d"%(turn))
            print(self)
            for p in self.players:
                print(str(p))
            for player_id in range(4):
                status = self.game_status()

                if status[0] == 'won':
                    print('Player %d won on turn %d !!!!!'%(status[1],turn))
                    return status[1]

                act = sample_random_action()
                trials = 0
                while act!=None and not self.action_is_valid_for_game(player_id, act):
                    act = sample_random_action()
                    trials += 1
                    if trials > 120:
                        print("ERROR: player %d could not find an action on turn %d. skipping."%(player_id,turn))
                        act = None

                if act:
                    self.execute_action_for_game(player_id, act)
        print("Game played to 150 turns, nobody won, probably error")
        return -1

    def game_status(self):
        winners = [pid for (pid,p) in enumerate(self.players) if p.points>=15]
        if len(winners) > 0:
            return ('won', winners[0])
        else:
            return ('in-progress',)


    def play_many_games_ai(self, games=500):
        for game in range(games):
            self.model.zero_grad()
            ret = self.play_entire_game_ai(log=False)
            if ret != 'draw':
                winner,turns = ret
                norm2 = sum((torch.norm(p) for p in self.model.parameters())).data[0]
                print("game ",game,"(",turns," turns) norm after ",norm2)
            self.reset()


    def play_entire_game_ai(self,log=True):
        for turn in range(200):
            if log:
                print("\nTURN",turn,)
                print(str(self),"\n")
            for pid in range(4):
                self.active_player = pid
                act = self.players[pid].find_action_with_model(self,self.model)

                self.execute_action_for_game(pid, act,log=log)

                status = self.game_status()
                if status[0] == 'won':
                    if log:
                        print('Player %d won on turn %d !!!!!'%(status[1],turn))
                        print("Optimizing ...")

                    self.players[status[1]].apply_reward(self.model, did_win=True)
                    for i in range(self.num_players):
                        if status[1] != i:
                            self.players[i].apply_reward(self.model, did_win=False)


                    return status[1], turn
        return 'draw'


    # let the AI make a choice and follow it
    def step_with_model(self):
        pid = self.active_player

        # See which cards can be picked-up
        cards = [kv[1] for kv in self.cards.items() if self.action_is_valid_for_game(pid,
                    Action(type='card',card_id=kv[0],coins=-1))]

        # The ordering of cst is important since we pass no indexing information
        # to our model. We will match card by data and find that way.
        gst,cst = self.to_features(cards)

        y = self.model(gst,cst)
        yy = y.numpy()
        
        # Map model's choice to an action
        act = None
        while act is None or not self.action_is_valid_for_game(pid,act):
            act_id = yy.argmax()
            yy[act_id] = -1
            if act_id < 10:
                act = Action('coins', card_id=-1, coins=one_hot(COIN_PICKUPS[act_id]))
            elif act_id < 15:
                act = Action('coins', card_id=-1, coins=one_hot(act_id-10)*2)
            else:
                card_id = cards[act_id-15]
                act = Action('card', card_id=card_id, coins=-1)


        ' Store both the action description and the softmax variable '
        self.players[pid].act_history.append(act, y)

        self.execute_action_for_game(pid, act)
        self.active_player = (self.active_player+1) % self.num_players



    ''' Util '''

    ' Return a 2-tuple of <gst,cst>, see model.py for the definitions '
    def to_features(self, cards=None):
        if cards is None:
            cards = self.cards.values()
        cards = np.concatenate([card.to_features() for card in cards])
        players = np.concatenate([player.to_features() for player in self.players])
        return np.concatenate([self.bank,players]), cards

    def __str__(self):
        return "Game bank=%s, cards=%d"%(str(self.bank), len([it for it in self.cards.items() if it[1] != None]))


def run_many_games():
    g = Game()
    g.play_many_games_ai(50000)

if __name__=='__main__' and 'train' in sys.argv:
    run_many_games()
