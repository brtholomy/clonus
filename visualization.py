import matplotlib.pyplot as plt


def PlotShow(plot=None):
    # special magic for avoiding a blocking call
    if plot:
        plt.show(plot, block=False)
    else:
        plt.show(block=False)
    # Pause to allow the input call to run:
    plt.pause(0.001)
    input("hit [enter] to end.")
    plt.close("all")
