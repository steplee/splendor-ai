# splendor-ai

pytorch based AI to help me win the board game Splendor :)

## Notes:
 - Policy Iteration is simply not working. Complete mode collapse. I think the game is to complex for the time+params I'm training it with, or you need a very stable initialization+good hyperparams.
 - Instead I am now working on an approach approximating the Value Function and examining all one-step future states (per-player, per-turn) and sampling one to move to. 
   - Since this requires simulating the game (in the sense you need to see future states without modifying the game state), I had to dirty-up game.py to do it and should go back and make it nicer.

## TODO:
 - [ ] __Do a rewrite of game.py completely imperatively using arrays, no objects!__ (need this for experience replay)
 - [ ] the code got messy when I implemented Value Iteration ... fix it up.
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
