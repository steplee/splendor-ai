import numpy as np
import random
import itertools,pdb

import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

#import game

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

        num_coin_pickups = 10+5+1 # 5 choose 3 + 5 + 1gold
        gst_size = 6+44+1 + 3
        cst_size = 6+5+1

        hg_sizes = [800, 600, 400, 300]
        hc_size = 300 # two layers.

        self.hg1 = nn.Linear(gst_size, hg_sizes[0])
        self.hg2 = nn.Linear(hg_sizes[0], hg_sizes[1])
        self.hg3 = nn.Linear(hg_sizes[1], hg_sizes[2])
        self.hg4 = nn.Linear(hg_sizes[2], hg_sizes[3])
        self.coin_scoring = nn.Linear(self.hg4.out_features, num_coin_pickups)

        self.hc1 = nn.Linear(cst_size, hc_size)
        self.hc2 = nn.Linear(hc_size, hc_size)
        
        ' We will concatenate the game_state and exactly 1 card_state '
        self.card_scoring = nn.Linear (self.hg4.out_features + self.hc2.out_features, 1)

        self.final_softmax = nn.Softmax()

        self.opt = torch.optim.SGD(self.parameters(), lr=.01)


    ' The part to only run once '
    ' Will return the hidden representation & the unnormalized coin-pickup scores '
    def gst_forward(self, gst):
        net = F.sigmoid(self.hg1(gst))
        net = F.sigmoid(self.hg2(net))
        net = F.sigmoid(self.hg3(net))
        net = F.sigmoid(self.hg4(net))
        coin_scores = F.sigmoid(self.coin_scoring(net))
        return (net,coin_scores)

    ' Run upto 12 times, return score '
    def cst_forward(self, gst_partial, cst):
        net = F.sigmoid(self.hc1(cst))
        net = F.sigmoid(self.hc2(net))
        #net = torch.cat([cst, net])
        net = torch.cat([gst_partial, net])
        #pdb.set_trace()
        score = F.sigmoid(self.card_scoring(net)) * .01
        return score

    ' Return softmax over all possible actions '
    def forward(self, gst,csts):
        if type(gst)==np.ndarray:
            gst = ag.Variable(torch.Tensor(gst), requires_grad=False)
            csts = [ag.Variable(torch.Tensor(cst), requires_grad=False) for cst in csts]

        gst_partial,coin_scores = self.gst_forward(gst)
        if len(csts) > 0:
            card_scores = [self.cst_forward(gst_partial, cst) for cst in csts]
            card_scores = torch.cat(card_scores)
            scores = torch.cat([coin_scores, card_scores])
        else:
            scores = coin_scores
        scores = scores.expand([1,len(scores)])
        scores = self.final_softmax(scores)

        #scores = scores.resize(scores.size()[1])
        #return list(sorted([(act,idx) for (idx,act) in enumerate(scores)], key=lambda ai:ai[0].data[0]))
        #return list([(act,idx) for (idx,act) in enumerate(scores)], key=lambda ai:ai[0].data[0])
        return scores


test_gst,test_cst = np.random.random(54),[np.random.random(12) for _ in range(6)]
tgst,tcsts = ag.Variable(torch.Tensor(test_gst)),[ag.Variable(torch.Tensor(arr)) for arr in test_cst]


'''
class SplendorModel(object):
    def __init__(self):
        self.output_size = 28
        self.net = Net(self.input_size, self.output_size)
        pass

    ' Returns a sorted list of tuple <score as torch variable, idx> '
    def predict_action(self, game):
        x = game.to_features()

        acts = self.net(x)
        acts = sorted([(act,idx) for (idx,act) in enumerate(y)])

        return acts
'''
