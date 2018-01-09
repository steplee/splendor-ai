# splendor-ai

pytorch based AI to help me win the board game Splendor :)

### v1.1:
Completely rewrote flow of the game logic. Instead of having a central game object, we have lightweight game states which are actually controlled by 'actors' (not in the traditional RL sense). They implement an `act` function which takes a state and gives a state. A game runner function in `game_harness.py` interacts with the actor instance to control the game flow and end it when needed.

Every game state is chained to the previous one in a `Record` node, which is a linked list ending with the winning state and shared between all players.

Implement different RL techniques by creating new actors. Actors also define how to apply rewards and the loss (which is seperate from the model entirely).

Right now, my Value Iteration actor encourages moves along winning paths to be 30.0 and losing paths to be near 0. It applies rewards at the end of every game, although
an **experience replay** approach would be possible.



## Notes:
  - Since the rewrite, I will need to implement Policy Gradients over again.

## TODO:
 - [ ] see if I can replicate+parallelize a subnet (you can definiately do it in TF)
 - [ ] automate hyperparam search.
 - [ ] I am wasting time simulating games and not operating on permuatations of players wrt. the game state. I should add permutations of players to the minibatch, since they should not change the choice
 - [x] profile and optimize using `python-flamegraph`
     - Most time is spent in pytorch ... which is what I wanted. But I need to rewrite w/ batching
 - [ ] pit different policied players against eachother, make sure mine can beat random, etc.
 - [ ] build web UI and Flask interface

##### Profiling
```
sudo pip3 install flamegraph

git clone https://github.com/brendangregg/FlameGraph /opt

python -m flamegraph -o perf.log <script.py> <args>
/opt/FlameGraph/flamegraph.pl -- title "<title>" perf.log > perf.svg

google-chrome perf.svg
```
