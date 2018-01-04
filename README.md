# splendor-ai

pytorch based AI to help me win the board game Splendor :)

## TODO:
 - [ ] move contents of model.py into policies/pg_player.py
 - [x] profile and optimize using `python-flamegraph`
     - Most time is spent in pytorch ... which is what I wanted. But I need to rewrite w/ batching
 - [ ] pit different policied players against eachother, make sure mine can beat random, etc.
 - [ ] tune hyperparameters
 - [ ] build web UI and Flask interface

##### Profiling
```
sudo pip3 install flamegraph

git clone https://github.com/brendangregg/FlameGraph /opt

python -m flamegraph -o perf.log <script.py> <args>
/opt/FlameGraph/flamegraph.pl -- title "<title>" perf.log > perf.svg

google-chrome perf.svg
```
