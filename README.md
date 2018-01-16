# splendor-ai

pytorch based AI to help me win the board game Splendor :)

### v1.2:
ValueIterationActor is not learning. Well actually it is... learning to lose. It learns to greedily selects cards in all cases. I think I need to do a several-step lookahead and not the one-step that is done now.

PolicyGradientPlayer wins more than average.

### v1.1:
Completely rewrote flow of the game logic. Instead of having a central game object, we have lightweight game states which are actually controlled by 'actors'. They implement an `act` function which takes a state and gives a state. A game runner function in `game_harness.py` interacts with the actor instance to control the game flow and end it when needed.

Every game state is chained to the previous one in a `Record` node, which is a linked list ending with the winning state and shared between all players. It is exposed to the actor in `apply_reward`.

Implement different RL techniques by creating new actors. Actors also define how to apply rewards and the loss.

Right now, my Value Iteration actor encourages moves along winning paths to be 30.0 and losing paths to be near 0. It applies rewards at the end of every game, although an **experience replay** approach would be possible.



## TODO:
 - [ ] test SGD / Adam, size of nets.
 - [ ] see if I can replicate+parallelize a subnet (you can definiately do it in TF)
 - [ ] **automate hyperparam search**.
 - [ ] I could permute user and card information and train on same games.
 - [x] profile and optimize using `python-flamegraph`
     - Most time is spent in pytorch ... which is what I wanted.
 - [ ] pit different policied players against eachother, make sure mine can beat random, etc.
 - [ ] build web UI, Flask interface, and JSON serialization

##### Profiling
```
sudo pip3 install flamegraph

git clone https://github.com/brendangregg/FlameGraph /opt

python -m flamegraph -o perf.log <script.py> <args>
/opt/FlameGraph/flamegraph.pl -- title "<title>" perf.log > perf.svg

google-chrome perf.svg
```
