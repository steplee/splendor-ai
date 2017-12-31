import numpy as np
import random
import itertools,pdb

import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

import game

def one_hot(i,size=5):
    a = np.zeros(size)
    if type(i) == int:
        a[i] = 1
    else:
        for j in i:
            a[int(j)] = 1
    return a


'''
gst: game state (all of bank, player info, and current player)
cst: card state (card attribs)
These are both handed over to our model at decsion time.

 - We need to have a softmax component per card, per card, there will be 12 by default but we only bother
   including those in the softmax that we can afford to buy.
 - A portion of the model need only be run once per decision outside of the per-card subnet, so
   this is a small optizmization.
 - This subnet will also output the unnormalized scores of picking up any combination of coins
 - The final softmax will have variable number of components since we can afford different cards at different times & the bank might not allow something
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_coin_pickups = 10+5 # 5 choose 3 + 5
        gst_size = 6+44+1
        cst_size = 6+5+1

        hg_sizes = [100, 90]
        hc_size = 100

        self.hg1 = nn.Linear(gst_size, hg_sizes[0])
        self.hg2 = nn.Linear(hg_sizes[0], hg_sizes[1])
        self.coin_scoring = nn.Linear(self.hg2.out_features, num_coin_pickups)

        self.hc1 = nn.Linear(cst_size, hc_size)
        
        ' We will concatenate the game_state and card_state '
        self.card_scoring = nn.Linear (self.hg2.out_features + self.hc1.out_features, 1)

        self.final_softmax = nn.Softmax()


    ' The part to only run once '
    ' Will return the hidden representation & the unnormalized coin-pickup scores '
    def gst_forward(self, gst):
        net = F.relu(self.hg1(gst))
        net = F.relu(self.hg2(net))
        coin_scores = F.relu(self.coin_scoring(net))
        return (net,coin_scores)

    ' Run upto 12 times, return score '
    def cst_forward(self, gst_partial, cst):
        net = F.relu(self.hc1(cst))
        #net = torch.cat([cst, net])
        net = torch.cat([gst_partial, net])
        #pdb.set_trace()
        score = F.relu(self.card_scoring(net))
        return score

    ' Return softmax over all possible actions '
    def forward(self, gst,csts):
        gst_partial,coin_scores = self.gst_forward(gst)
        card_scores = [self.cst_forward(gst_partial, cst) for cst in csts]
        card_scores = torch.cat(card_scores)

        scores = torch.cat([coin_scores, card_scores])
        scores = scores.expand([1,len(scores)])
        scores = self.final_softmax(scores)
        return scores.resize(scores.size()[1])

test_gst,test_cst = np.random.random(51),[np.random.random(12) for _ in range(6)]
tgst,tcsts = ag.Variable(torch.Tensor(test_gst)),[ag.Variable(torch.Tensor(arr)) for arr in test_cst]


class SplendorModel(object):
    def __init__(self):
        self.output_size = 28
        self.net = Net(self.input_size, self.output_size)
        pass

    # state -> softmax_over_actions
    def predict_action(self, game):
        x = game.to_features()

        acts = self.net(x)
        acts = sorted([(act,idx) for (idx,act) in enumerate(y)])

        return 
