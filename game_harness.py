from util import Record, Actions

from actors import ValueIterationActor
from models import ValueIterationModel

from game_logic import GameState

import os,sys,time,optparse

import torch



optp = optparse.OptionParser()
optp.add_option('--name', default=str(time.time()).split('.')[0])
optp.add_option('--policy_gradients', action="store_true")
optp.add_option('--load_model', action="store_true")
optp.add_option('--lr', default=.04, type='float')
optp.add_option('--save_every', default=400, type='float')
optp.add_option('--test_every', default=100, type='float')
optp.add_option('--sgd', action='store_true') # Use sgd instead of adam
opts = optp.parse_args(sys.argv)[0]



def play_some_games(n=5000):
    via = ValueIterationActor.ValueIterationActor()
    save_path = os.path.join('saved_models',opts.name)
    log_path = os.path.join('logs',opts.name)

    with open(log_path,'w') as logf:
        if opts.load_model:
            pass # TODO
        else:
            opt_method = 'sgd' if opts.sgd else 'adam'
            model = ValueIterationModel.Net(lr=opts.lr, opt_method='sgd')

        for game_num in range(n):
            genesis_state = GameState()
            genesis_record = Record(None,GameState(),None,None,model,'genesis',-1)
            record = genesis_record
            stat = ('ongoing',-1)
            turns = 0
            #print("Start game",i)

            while stat[0] == 'ongoing':
                stat,record = via.act(record, model)
                turns += 1

                if turns%4==0 and turns>0:
                    pass
                    #print("(turn",str(turns//4)+")")
                    #time.sleep(3)

                if stat[0] == 'fail':
                    print("Game",game_num,"failed!, doing nothing")
                    print("Game",game_num,"failed!, doing nothing",file=logf)
                    break
                elif stat[0] == 'draw':
                    print("Game",game_num,"was a draw, doing nothing")
                    print("Game",game_num,"was a draw, doing nothing",file=logf)
                elif stat[0] == 'won':
                    if game_num % 5 == 0:
                        print("Player {} won game {} on round {}".format(record.status,game_num,turns//4))
                        print("Player {} won game {} on round {}".format(record.status,game_num,turns//4),file=logf)
                    via.apply_reward(record)

            if game_num % opts.save_every == 0 and game_num>0:
                print("\nsaving to", save_path)
                torch.save({'opts': opts,'game_num':game_num, 'state_dict':model.state_dict} , save_path)

            if game_num % opts.test_every == 0 and game_num>0:
                print("\nTesting:")
                turn_cnt = []
                for test_num in range(6):
                    genesis_state = GameState()
                    genesis_record = Record(None,GameState(),None,None,model,'genesis',-1)
                    record = genesis_record
                    stat = ('ongoing',-1)
                    test_turns = 0
                    while stat[0] == 'ongoing':
                        stat,record = via.act(record, model, training=False)

                        if stat[0] == 'fail' or stat[0] == 'draw':
                            pass
                        elif stat[0] == 'won':
                            turn_cnt.append(test_turns//4)
                        test_turns += 1
                print("Average turns over %d tests:"%len(turn_cnt),sum(turn_cnt)/len(turn_cnt),'\n')
                print("Average turns over %d tests:"%len(turn_cnt),sum(turn_cnt)/len(turn_cnt),'\n',file=logf)
                logf.flush()



if __name__=='__main__' and 'train' in sys.argv:
    play_some_games(10000)
