import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load obtained data
pickle_in = open("./data/experiment_result.pickle", "rb")
MAIN = pickle.load(pickle_in)
pickle_in.close()

# Select data for which to plot the heatmaps (hinge, basic, ratio)
config = "basic"
batch_set = ['16', '32', '64', '128', '256', '512']
batch_dict = {16: 0, 32: 1, 64: 2, 128: 3, 256: 4, 512: 5}
for run in range(10):
    batch_size = len(batch_set)
    data = np.zeros((batch_size, 100))
    batches = []
    for i in range(100):
        data[:, i] = MAIN['RMGD'][config][run]["probability"][i]
        batches.append(batch_dict[MAIN['RMGD'][config][run]["batch_size"][i]])

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(data, vmin=0, vmax=1, cmap='jet')
    ax.set_yticks(np.arange(len(batch_set)))
    ax.set_yticklabels(batch_set)

    # Add dot for selected batch size at each epoch
    for i in range(100):
        chosen = MAIN['RMGD'][config][run]["batch_size"][i]
        for j in range(len(batch_set)):
            if batch_set[j] == str(chosen):
                symb = "o"
                for font in range(1, 15):
                    if font > 12:
                        text = ax.text(i, j, symb, ha="center", va="center", color="k", fontsize=font)
                    else:
                        text = ax.text(i, j, symb, ha="center", va="center", color="w", fontsize=font)

    cbar = ax.figure.colorbar(im, ax=ax, cmap='jet', shrink=0.05)
    fig.tight_layout()
    plt.savefig('./image/' + config + '_run' + str(run) + '.eps', bbox_inches='tight')

