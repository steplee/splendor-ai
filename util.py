import numpy as np
import itertools
from collections import namedtuple

def one_hot(i,size=5):
    a = np.zeros(size)
    if type(i) == int or type(i)==np.int64:
        a[i] = 1
    else:
        for j in i:
            a[int(j)] = 1
    return a

def one_neg(i,size=5):
    a = np.ones(size) * -1
    if type(i) == int or type(i)==np.int64:
        a[i] = 1
    else:
        for j in i:
            a[int(j)] = 1
    return a


''' - -  -  -  -  -  -  RECORD  -  -  -  -  -  -  -  -  -  -  '''

Record = namedtuple('record', ['prev','state','values_var','act_id','model','status','pid'])




''' - -  -  -  -  -  -  COINS & CARDS  -  -  -  -  -  -  -  -  -  -  '''

' The ways to pickup 3 coins, encoded as three-hot vectors '
COIN_PICKUPS = (itertools.combinations(range(5),3))
COIN_PICKUPS = list(map(one_hot, COIN_PICKUPS))
COIN_PICKUPS = [np.concatenate([x,[0]]) for x in COIN_PICKUPS]

#LEVEL_NUM_CARDS = [43, 20, 15]

# Each level is list of tuple <weight, cost, points>
# Since colors are exchangeable, the `cost` is color independent and generated randomly
LEVEL_TEMPLATES = [
        [ (1,[1,1,1,1],0), (1,[1,1,1],0), (.2,[2,2,1],1) ],
        [ (1,[2,2,3],2), (1,[3,3,2],2), (.2,[6],3) ],
        [ (1,[7],4), (1,[3,3,4],3), (1,[4,4,3],4) ]
]

LEVEL_PROBABILITIES = [[],[],[]]
# Normalize
for lvl in range(3):
    row_sum = sum([t[0] for t in LEVEL_TEMPLATES[lvl]])
    for j,t in enumerate(LEVEL_TEMPLATES[lvl]):
        LEVEL_PROBABILITIES[lvl].append( t[0]/row_sum )


''' - -  -  -  -  -  -  ACTION  -  -  -  -  -  -  -  -  -  -  '''

#                               str      int       array
Action = namedtuple('Action',['type', 'card_id', 'coins'])
Actions = [Action('coins', card_id=-1, coins=(COIN_PICKUPS[i])) for i in range(10)] \
        + [Action('coins', card_id=-1, coins=one_hot(i,6)*2) for i in range(5)] \
        + [Action('reserve', card_id=(np.random.randint(3)*4+np.random.randint(3)),coins=-1)] \
        + [Action('card', card_id=(i*4+j), coins=-1) for i in range(3) for j in range(4)]

Active_Players = [one_hot(i,size=4) for i in range(4)] # performance opt
