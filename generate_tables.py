import pickle
import numpy as np
from tabulate import tabulate

# Load data from memory
pickle_in = open("./data/experiment_result.pickle", "rb")
MAIN_FULL = pickle.load(pickle_in)
pickle_in.close()

# Definition of variables
batch_list = ['16', '32', '64', '128', '256', '512']
CONFIGS = [["basic", 0, 0], ["sub", 1, 0], ["super", 2, 0], ["hinge", 0, 1], ["ratio", 0, 2]]
runs = 10
epochs = 100
chop = epochs - 1
array = [["Algorithms", "Mean pm SD", "Max", "Min"]]

# Generate table for MGD
for i, batch in enumerate(batch_list):
    print("-"*60)
    print("-"*60)
    print("-"*60)
    tmp = np.zeros((runs, epochs - chop))
    for run in range(runs):
        try:
            print(MAIN_FULL['MGD'][batch][run]["test_accuracy"])
            tmp[run, :] = np.array(MAIN_FULL['MGD'][batch][run]["test_accuracy"][chop:])
        except KeyError:
            # we don't have the data yet
            print("NO DATA")

    tmp.reshape((runs * (epochs - chop), 1))
    mean = np.mean(tmp)
    std = np.std(tmp)
    max_acc = np.max(tmp)
    min_acc = np.min(tmp)

    lst = ["MGD (" + str(batch) + ")",
            "{0:.3f}".format(round(mean, 3)) + " pm " + "{0:.3f}".format(round(std, 3)),
            "{0:.3f}".format(round(max_acc, 3)),
            "{0:.3f}".format(round(min_acc, 3))]
    array.append(lst)

# Generate table for RMGD
batch_list = [CONFIGS[0][0], CONFIGS[1][0], CONFIGS[2][0], CONFIGS[3][0], CONFIGS[4][0]]
for i, batch in enumerate(batch_list):
    print("-"*60)
    print("-"*60)
    print("-"*60)
    tmp = np.zeros((runs, epochs - chop))
    for run in range(runs):
        try:
            print(CONFIGS[i][0])
            print(MAIN_FULL['RMGD'][batch][run]["test_accuracy"])
            tmp[run, :] = np.array(MAIN_FULL['RMGD'][batch][run]["test_accuracy"][chop:])

        except KeyError:
            # we don't have the data yet
            print("NO DATA")

    tmp.reshape((runs * (epochs - chop), 1))
    mean = np.mean(tmp)
    std = np.std(tmp)
    max_acc = np.max(tmp)
    min_acc = np.min(tmp)

    lst = ["RMGD (" + str(batch) + ")",
           "{0:.3f}".format(round(mean, 3)) + " pm " + "{0:.3f}".format(round(std, 3)),
           "{0:.3f}".format(round(max_acc, 3)),
           "{0:.3f}".format(round(min_acc, 3))]
    array.append(lst)

table_string = (tabulate(array, tablefmt="latex", floatfmt=".2f"))
table_string = table_string.replace('&', '$&$')
table_string = table_string.replace('\\\\', '\\\\ \\hline')
table_string = table_string.replace('pm', '\\pm')
table_string = table_string.replace(' MGD', '$MGD')
table_string = table_string.replace(' RMGD', '$RMGD')

# Paste this in latex
print(table_string)

