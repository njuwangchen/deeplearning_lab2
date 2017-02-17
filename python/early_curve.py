import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_early_curve(filename):
    training_accuracy = []
    tuning_accuracy = []
    testing_accuracy = []

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Training"):
                training_accuracy.append(line.split()[3])
            elif line.startswith("Tuning"):
                tuning_accuracy.append(line.split()[3])
            elif line.startswith("Testing"):
                testing_accuracy.append(line.split()[3])
            else:
                continue

    # plot starts

    fig, ax = plt.subplots()

    fig.set_size_inches(12, 8)

    plt.xlabel("epoch num")
    plt.ylabel("accuracy")

    title_arr = filename.split('_')
    hidden_units = title_arr[0]
    learning_rate = title_arr[1]
    momentum = title_arr[2]
    weight_decay = title_arr[3]
    activation = title_arr[4]

    title = "Hidden units: "+hidden_units+" Learning rate: "+learning_rate+\
            "\nMomentum: "+momentum+" Weight Decay: "+weight_decay+\
            "\nHidden Layers Activation: "+activation

    plt.title(title)

    ax.plot(training_accuracy, "r-", label = "training accuracy")
    ax.plot(tuning_accuracy, "g-", label = "tuning accuracy")
    ax.plot(testing_accuracy, "b-", label = "testing accuracy")

    ax.legend(loc = 'upper left', shadow=True)

    x_major_locator = MultipleLocator(5)
    x_major_formatter = FormatStrFormatter("%d")
    x_minor_locator = MultipleLocator(1)

    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_major_formatter(x_major_formatter)

    ax.xaxis.set_minor_locator(x_minor_locator)

    y_major_locator = MultipleLocator(0.02)
    y_minor_locator = MultipleLocator(0.01)

    ax.yaxis.set_major_locator(y_major_locator)

    ax.yaxis.set_minor_locator(y_minor_locator)

    ax.grid(b=True, which='major', color='0.65', linestyle='-')
    ax.grid(b=True, which='minor', color='0.2', linestyle='-.')

    ind_of_highest_tune = len(training_accuracy)-1-11
    final_training = float(training_accuracy[ind_of_highest_tune])
    final_tuning = float(tuning_accuracy[ind_of_highest_tune])
    final_testing = float(testing_accuracy[ind_of_highest_tune])

    ax.annotate("training accuracy:\n" + str(final_training),
                 xy=(ind_of_highest_tune, final_training),
                 xytext=(ind_of_highest_tune, final_training - 0.01))
    ax.annotate("tuning accuracy:\n" + str(final_tuning),
                 xy=(ind_of_highest_tune, final_tuning),
                 xytext=(ind_of_highest_tune, final_tuning - 0.01))
    ax.annotate("testing accuracy:\n"+str(final_testing),
                 xy=(ind_of_highest_tune, final_testing),
                 xytext=(ind_of_highest_tune, final_testing - 0.01))

    ax.axvline(x=ind_of_highest_tune, color='m', linewidth=2)

    back = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if back == 'TkAgg':
        mng.resize(*mng.window.maxsize())

    plt.savefig(filename+".png")

if __name__ == '__main__':
    plot_early_curve("10_0.01_0.9_0_sigmoid")
