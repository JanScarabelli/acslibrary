import string

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(df, var: string, target: string, bins=20, palette="coolwarm", alpha=1):
    """
    This function receives two variables as input (var and target), the histogram of the first one is plotted,
    for every bin of the histogram, the mean of the second variable in the corresponding bin is printed. A colormap
    corresponding to the mean in each bin is applied later.

    df: Panda data frame

    var: string
    name of the variable on wich the histogram is created

    target: string
    name of the variable used to evaluate the bin (normally yield)

    bins:
    number of bins of the histogram

    palette: string
    name of a matplotlib colormap

    alpha: [0,1]
    variable of matplotlib.hist that controls color saturation

    """
    # Set the figure size
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams["figure.autolayout"] = True

    # Figure and set of subplots
    fig, ax = plt.subplots()

    # Call plt.hist and saving variables
    counts, bins, patches = ax.hist(
        df[var], edgecolor='black', linewidth=0.05, bins=bins, alpha=alpha
    )

    # Calculating yield mean for each bin#
    yields = []
    yield_mean = 0
    k = 0
    minimum = df[target].max()

    # The last bin is done outside the loop
    for i in range(0, len(counts) - 1):

        for j in range(0, len(df)):

            if (df.loc[j, var] >= bins[i]) and (df.loc[j, var] < bins[i + 1]):
                k = k + 1
                yield_mean += df.loc[j, target]

        if k != 0:
            yield_mean = yield_mean / k
        else:
            yield_mean = 0

        if (yield_mean < minimum) and (yield_mean != 0):
            minimum = yield_mean

        yields.append(yield_mean)
        k = 0
        yield_mean = 0

    # The last bin done apart
    for j in range(0, len(df)):

        if (df.loc[j, var] >= bins[len(counts) - 1]) and (df.loc[j, var] <= bins[len(counts)]):
            k = k + 1
            yield_mean += df.loc[j, target]

    if k != 0:
        yield_mean = yield_mean / k
    else:
        yield_mean = 0

    if (yield_mean < minimum) and (yield_mean != 0):
        minimum = yield_mean

    yields.append(yield_mean)

    # Computing max of yields
    yields = np.array(yields)
    maximum = np.max(yields)

    # Scaling yield means for the colormap
    yields = (yields - minimum) / (maximum - minimum)

    # Defining colormap
    cmap = plt.colormaps[palette]

    # Assigning a color from the colormap to each bin in function of yield mean in that bin
    for i in range(len(counts)):
        patches[i].set_facecolor(cmap(yields[i]))

    # Rescaling yield to the original values
    yields = (yields * (maximum - minimum)) + minimum

    # Printing yield means in top of bins histogram
    for patch, label in zip(patches, yields):
        height = patch.get_height()
        if height > 0:  # Skips the print of the yield mean in the variables with frequency 0
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.02,
                "%.2f" % label,
                ha='center',
                va='bottom',
            )

    # Display the plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    plt.title('Histogram for variable ' + var + ' with target ' + target)
    plt.show()
