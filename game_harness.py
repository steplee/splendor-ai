from util import Record, Actions

from actors import ValueIterationActor
from models import ValueIterationModel

from game2 import GameState

import sys,time

def play_some_games(n=5000):
    via = ValueIterationActor.ValueIterationActor()
    model = ValueIterationModel.Net()

    for i in range(n):
        genesis_state = GameState()
        genesis_record = Record(None,GameState(),None,None,model,'genesis',-1)
        record = genesis_record
        stat = ('ongoing',-1)
        turns = 0
        print("Start game",i)

        while stat[0] == 'ongoing':
            stat,record = via.act(record, model)
            turns += 1

            if turns%4==0 and turns>0:
                print("(turn",str(turns//4)+")")
                #time.sleep(3)

            if stat[0] == 'draw':
                print("Game",i,"was a draw, doing nothing")
            elif stat[0] == 'won':
                print("Player {} won game {} on round {}".format(record.status,i,turns//4))
                via.apply_reward(record)


if __name__=='__main__' and 'train' in sys.argv:
    play_some_games(10000)
