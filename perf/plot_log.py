import re,sys
import numpy as np
from matplotlib import pyplot as plt

def plot_log(fname="log"):
    turns = []

    name2id = {'RandomActor':0,'PolicyIterationActor':1,'ValueIterationActor':2}

    if 'wins' in sys.argv:
        with open(fname,'r') as f:
            wins = []
            names = []
            for line in f.readlines():
                if "freqs" in line and "[('" in line:
                    lst = eval(re.search('(\[\(.*\)\])',line).groups()[0])
                    lst = sorted(lst)
                    if len(names)==0: names = [p for (p,w) in lst]
                    wins.append([float(w) for (p,w) in lst])
        wins = np.array(wins).T
        for (name,win) in zip(names,wins):
            plt.plot(win, label=name)
        plt.legend()
        plt.show()

    elif 'test' not in sys.argv:
        with open(fname,'r') as f:
            for line in f.readlines():
                if ' on round' in line:
                    turns.append(int(line.rsplit(' ')[-1]))
        plt.plot(turns)
        plt.show()
    else:
        with open(fname,'r') as f:
            for line in f.readlines():
                if ' tests' in line:
                    turns.append(float(line.strip().rsplit(' ')[-1]))
        plt.plot(turns)
        plt.show()


if __name__=='__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = 'log'
    plot_log(fname)
