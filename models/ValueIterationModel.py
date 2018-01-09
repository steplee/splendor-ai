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

use_dropout = False
#use_dropout = True

# Change the non-linearity by setting this
f,ff = F.relu, torch.nn.ReLU
#f,ff = F.sigmoid, torch.nn.Sigmoid
#f = F.sigmoid
#f = F.tanh

'''
Architecture:

          value approx.
              |
              |
        join all cards+hg
             ____
            /     \
           |       \
           |        \
           |         |
           |         |
           |        / \
           |      /     \
           |    /        |
           |   |      (hc subnet)    <--- 12 repetitions
        (hg subnet)        |
            |              |
            |              |
        game_state       [card]
'''

class Net(nn.Module):
    def __init__(self, lr=.04):
        super(Net, self).__init__()

        num_coin_pickups = 10+5+1 # 5 choose 3 + 5 + 1gold
        gst_size = 6+52+4
        cst_size = 6+6+1

        #hg_sizes = [400, 300, 300]
        #hc_sizes = [200, 150, 100]
        hg_sizes = [900,800,600]
        hc_sizes = [200, 120]

        # Game state subnet
        self.hg = []
        for i in range(len(hg_sizes)):
            if i == 0:
                self.hg.append( nn.Linear(gst_size, hg_sizes[0]) )
                self.hg.append( ff() )
            else:
                self.hg.append( nn.Linear(hg_sizes[i-1], hg_sizes[i]) )
                self.hg.append( ff() )

        if use_dropout:
            self.hg_dropout = nn.Dropout(.25)
            self.hg_seq = nn.Sequential( *(self.hg + [self.hc_dropout]) )
        else:
            self.hg_seq = nn.Sequential(*self.hg)

        # Card subnet
        self.hc = []
        for i in range(len(hc_sizes)):
            if i == 0:
                self.hc.append( nn.Linear(cst_size, hc_sizes[0]) )
                self.hc.append( ff() )
            else:
                self.hc.append( nn.Linear(hc_sizes[i-1], hc_sizes[i]) )
                self.hc.append( ff() )

        if use_dropout:
            self.hc_dropout = nn.Dropout(.25)
            self.hc_seq = nn.Sequential( *(self.hc + [self.hc_dropout]) )
        else:
            self.hc_seq = nn.Sequential(*self.hc)
        
        ' We will concatenate the game_state activations and all 12 card_states '
        self.final_scoring = nn.Linear(self.hg_seq[-2].out_features + 12*self.hc[-2].out_features, 1)
        self.final_softmax = nn.Softmax()

        #self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=.00001)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=.00005)


    ' The part to only run once '
    ' Will return the hidden representation & the unnormalized coin-pickup scores '
    def gst_forward(self, gst):
        net = self.hg_seq(gst)
        return net

    ' Run 12 times, return hidden rep '
    def cst_forward(self, gst_partial, cst):
        net = self.hc_seq(cst)
        return net

    ' Return value over single game state '
    def forward(self, gst,csts):
        if type(gst)==np.ndarray or type(gst)==list:
            gst = ag.Variable(torch.FloatTensor(gst), requires_grad=False)
            csts = ag.Variable(torch.FloatTensor(csts), requires_grad=False)

        gst_partial = self.gst_forward(gst)

        #assert(len(csts) == 12)

        # batch x 20
        #card_reps = [self.cst_forward(gst_partial, cst).view(-1) for cst in csts]
        #print('cr',card_reps[0].size())
        #card_reps = [torch.cat([joined[i]]+[card_reps[i]]) for i in range(16)]

        ''' FOR VALUE '''
        ' Batch by 28 future states '
        # gst_partial : 28 x 1000
        cr = self.cst_forward(gst_partial, csts).view(28, -1)

        #cr = torch.cat([gst_partial.expand([28,600,
        cr = torch.cat([gst_partial,cr],dim=1)
        #cr[cr<0.00] = 0.000001

        scores = self.final_scoring(cr)
        scores= torch.clamp(scores, .000000001,100)


        ''' FOR SOFTMAX OVER ACTIONS
        cr = self.cst_forward(gst_partial, csts).view(28, -1)
        print('gstp',gst_partial.size())
        print('cr',cr.size())
        jo = torch.cat([gst_partial,cr], dim=1)
        print('jo',jo.size())

        scores = self.final_scoring(jo)
        print('scores_scorefinal',scores.size())
        scores = self.final_softmax(scores)
        print('scores_scoresoft',scores.size())
        '''

        '''
        ' Batched by many gsts'
        cr = self.cst_forward(gst_partial, csts).view(gst_partial.size()[0],-1)
        jo = torch.cat([gst_partial,cr], dim=1)

        scores = self.final_softmax(jo)
        '''

        #scores = F.relu(self.final_scoring(jo))
        #scores = f(self.final_scoring(jo))
        #scores[scores<0] = 0.00000001
        #scores = torch.clamp(scores, 0.000000001, 10000)

        return scores


#test_gst,test_cst = np.random.random(54),[np.random.random(12) for _ in range(6)]
#tgst,tcsts = ag.Variable(torch.Tensor(test_gst)),[ag.Variable(torch.Tensor(arr)) for arr in test_cst]

