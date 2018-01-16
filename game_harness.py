from util import Record, Actions

from actors import ValueIterationActor
from actors import RandomActor
from actors import PolicyGradientActor

from models import ValueIterationModel
from models import PolicyGradientModel

from game_logic import GameState

import os,sys,time,optparse
import numpy as np
import random

import torch



optp = optparse.OptionParser()
optp.add_option('--name', default=str(time.time()).split('.')[0])
optp.add_option('--policy_gradients', action="store_true")
optp.add_option('--load_model', action="store_true")
optp.add_option('--pg_lr', default=.04, type='float') # policy gradient lr
optp.add_option('--vi_lr', default=.04, type='float') # value iteration lr
optp.add_option('--save_every', default=800, type='float')
optp.add_option('--test_every', default=200, type='float')
optp.add_option('--sgd', action='store_true') # Use sgd instead of adam
opts = optp.parse_args(sys.argv)[0]



def play_some_games(n=5000):
    if opts.load_model:
        pass # TODO
    else:
        opt_method = 'sgd' if opts.sgd else 'adam'
        vi_model = ValueIterationModel.Net(lr=opts.vi_lr, opt_method=opt_method)
        vi2_model = ValueIterationModel.Net(lr=opts.vi_lr, opt_method=opt_method)
        pg_model = PolicyGradientModel.Net(lr=opts.pg_lr, opt_method=opt_method)


    via = ValueIterationActor.ValueIterationActor(use_model=vi_model)
    via2 = ValueIterationActor.ValueIterationActor(use_model=vi2_model)
    ra = RandomActor.RandomActor()
    pga = PolicyGradientActor.PolicyGradientActor(use_model=pg_model)

    all_save_dict = {'vi_dict': vi_model.state_dict, 'pg_dict': pg_model.state_dict}

    ' These are called `act(record)` upon '
    actors = [via, via2, pga, ra]
    actor_map = {a:0 for a in actors}


    save_path = os.path.join('saved_models',opts.name)
    log_path = os.path.join('logs',opts.name)

    with open(log_path,'w') as logf:

        for game_num in range(n):
            ' Permute players so that models generalize '
            #random.shuffle(actors)

            genesis_state = GameState()
            genesis_record = Record(None,GameState(),None,None,None,'genesis',-1)
            record = genesis_record
            stat = ('ongoing',-1)
            turns = 0
            #print("Start game",i)

            while stat[0] == 'ongoing':
                stat,record = actors[record.state.active_player].act(record,training=True)
                turns += 1

                if stat[0] == 'fail':
                    print("Game",game_num,"failed!, doing nothing")
                    print("Game",game_num,"failed!, doing nothing",file=logf)
                    break
                elif stat[0] == 'draw':
                    print("Game",game_num,"was a draw, doing nothing")
                    print("Game",game_num,"was a draw, doing nothing",file=logf)
                elif stat[0] == 'won':
                    if game_num % 5 == 0:
                        print("Player {} won game {} on round {}".format(actors[stat[1]].name,game_num,turns//4))
                        print("Player {} won game {} on round {}".format(actors[stat[1]].name,game_num,turns//4),file=logf)

                    for actor in actor_map.keys():
                        actor.apply_reward(record)

            if game_num % opts.save_every == 0 and game_num>0:
                print("\nsaving to", save_path)
                torch.save({'opts': opts,'game_num':game_num, **all_save_dict} , save_path)



            ' Testing code '

            if game_num % opts.test_every == 0 and game_num>0:
                print("\nTesting:")
                turn_cnt = []
                for test_num in range(5):
                    genesis_state = GameState()
                    genesis_record = Record(None,GameState(),None,None,None,'genesis',-1)
                    record = genesis_record
                    stat = ('ongoing',-1)
                    test_turns = 0
                    while stat[0] == 'ongoing':
                        #stat,record = actors[record.state.active_player].act(record,training=False)
                        stat,record = actors[record.state.active_player].act(record,training=True)

                        if stat[0] == 'fail' or stat[0] == 'draw':
                            pass
                        elif stat[0] == 'won':
                            actor_map[actors[record.state.active_player]] += 1
                            turn_cnt.append(test_turns//4)
                        test_turns += 1
                print("Average turns over %d tests:"%len(turn_cnt),sum(turn_cnt)/len(turn_cnt))
                print("Average turns over %d tests:"%len(turn_cnt),sum(turn_cnt)/len(turn_cnt),file=logf)
                print("Running player test win count,", [(a.name,w) for (a,w) in actor_map.items()])
                print("Running player test win count,", [(a.name,w) for (a,w) in actor_map.items()],file=logf)
                print("Running player test win freqs,", [(a.name,w/sum(actor_map.values())) for (a,w) in actor_map.items()],'\n')
                print("Running player test win freqs,", [(a.name,w/sum(actor_map.values())) for (a,w) in actor_map.items()],'\n',file=logf)
                logf.flush()



if __name__=='__main__' and 'train' in sys.argv:
    play_some_games(70000)
