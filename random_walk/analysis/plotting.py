import matplotlib.pyplot as plt
import numpy as np


def plot_msd(msd, save=None, show=True):
    plt.figure()
    plt.plot(msd)
    plt.xlabel("Step")
    plt.ylabel("Mean Squared Displacement")
    plt.title("MSD vs Step")
    plt.grid(True)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_energy(energies, save=None, show=True):
    plt.figure()
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy Minimisation")
    plt.grid(True)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
