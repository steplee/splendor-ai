# splendor-ai

pytorch based AI to help me win the board game Splendor :)

### v1.1:
Completely rewrote flow of the game logic. Instead of having a central game object, we have lightweight game states which are actually controlled by 'actors'. They implement an `act` function which takes a state and gives a state. A game runner function in `game_harness.py` interacts with the actor instance to control the game flow and end it when needed.

Every game state is chained to the previous one in a `Record` node, which is a linked list ending with the winning state and shared between all players. It is exposed to the actor in `apply_reward`.

Implement different RL techniques by creating new actors. Actors also define how to apply rewards and the loss.

Right now, my Value Iteration actor encourages moves along winning paths to be 30.0 and losing paths to be near 0. It applies rewards at the end of every game, although an **experience replay** approach would be possible.



## Notes:
  - Since the rewrite, I will need to implement Policy Gradients over again.
  - What happens, in value iteration, when we have a high-weight for an invalid state and never select it? If we win, we will change *the choice we did make*, but not the invalid one. This is important because I want to only call `forward` with valid states. Some solutions (that still require computing w/ invalid states):
    1. I could include the invalid states in the `forward` batch and encourage low values. The **model would actively learn the game's rules**
    2. I could include the invalid states and put a norm-penalty on all action values. In the long run the 'good' actions still get pushed up.

## TODO:

 - [ ] test SGD / Adam, size of nets.
 - [ ] see if I can replicate+parallelize a subnet (you can definiately do it in TF)
 - [ ] **automate hyperparam search**.
 - [ ] I could permute user and card information and train on same games.
 - [x] profile and optimize using `python-flamegraph`
     - Most time is spent in pytorch ... which is what I wanted.
     - Still I could batch by game, but it would complicate code a lot.
 - [x] pit different policied players against eachother, make sure mine can beat random, etc.
 - [ ] build web UI, Flask interface, and JSON serialization

##### Profiling
```
sudo pip3 install flamegraph

git clone https://github.com/brendangregg/FlameGraph /opt

python -m flamegraph -o perf.log <script.py> <args>
/opt/FlameGraph/flamegraph.pl -- title "<title>" perf.log > perf.svg

google-chrome perf.svg
```
