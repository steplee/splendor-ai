import re,sys
import numpy as np
from matplotlib import pyplot as plt

def plot_log(fname="log"):
    turns = []

    if 'test' not in sys.argv:
        with open(fname,'r') as f:
            for line in f.readlines():
                if ' on round' in line:
                    turns.append(int(line.rsplit(' ')[-1]))
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
