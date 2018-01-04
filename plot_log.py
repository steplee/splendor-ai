import re,sys
import numpy as np
from matplotlib import pyplot as plt

def plot_log(fname="log"):
    turns = []
    with open(fname,'r') as f:
        for line in f.readlines():
            if 'game' in line and 'turns' in line:
                x = int(re.match(".*\( (\d+).*\).*", line).groups()[0]) 
                turns.append(x)

    plt.plot(turns)
    plt.show()

if __name__=='__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = 'log'
    plot_log(fname)
