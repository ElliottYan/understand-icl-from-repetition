import matplotlib.pyplot as plt
import os, pickle
from analyze_self_reinforce import Stats

data_dir = 'data/'

def plot(lines, keys, title=""):
    # Create a figure and axis object
    fig, ax = plt.subplots()
    
    markers = ['o', '.', '*', '-']
    i = 0
    for x, y, z in lines:
        ax.errorbar(x, y, capsize=4, marker=markers[i], label=keys[i])
        i += 1

    # Set the title and axis labels
    ax.set_xlabel('# Repetition')
    ax.set_ylabel('TP: Average Token Prob')

    # Add a legend
    ax.legend()
    if title:
        ax.set_title(title)

    # Show the plot
    plt.show()

model_sigs = [
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
]

keys = ['wiki', 'random', 'book']


for ms in model_sigs:
    key = "TP" # WR, IP
    draw_data = []
    pkls = [os.path.join(data_dir, "{}_stats_{}.pkl".format(ms, key)) for key in keys]
    for fn in pkls:
        with open(fn, 'rb') as f:
            stats_dict = pickle.load(f)
        x = [i for i in stats_dict[key] if (i)%5==0]
        y = [stats_dict[key][i].get_mean() for i in stats_dict[key] if (i)%5==0]
        z = [stats_dict[key][i].get_variance() for i in stats_dict[key] if (i)%5==0]
        draw_data.append((x,y,z))
    plot(draw_data, keys, title=ms)
