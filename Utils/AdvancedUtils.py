import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pylab as pylab
from Utils.SignalUtils import *
from Utils.GeneralUtils import *
import torch

from mpl_toolkits import mplot3d


params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)


class AdvancedUtils:
    def __init__(self):
        return

    @staticmethod
    def f_theta_pi(r, theta, theta_p):
        return (1 - np.power(r, 2) * np.cos(2 * theta)) * (np.cos(theta_p)) - \
               (np.power(r, 2) * np.sin(2 * theta) * np.sin(theta_p))

    @staticmethod
    def f_min(r, theta):
        theta_p = np.arctan((np.power(r, 2) * np.sin(2 * theta)) / (np.power(r, 2) * np.cos(2 * theta) - 1))
        return AdvancedUtils.f_theta_pi(r, theta, theta_p)

    @staticmethod
    def f_max(r, theta):
        theta_p = np.arctan((np.power(r, 2) * np.sin(2 * theta)) / (np.power(r, 2) * np.cos(2 * theta) - 1))
        theta_p += np.pi
        return AdvancedUtils.f_theta_pi(r, theta, theta_p)

    @staticmethod
    def g_p1(r):
        return 1 / (1 - np.power(r, 2))

    @staticmethod
    def g_p2_min(r, theta):
        return AdvancedUtils.f_min(r, theta) / (1 + np.power(r, 4) - 2 * np.power(r, 2) * np.cos(2 * theta))

    @staticmethod
    def g_p2_max(r, theta):
        return AdvancedUtils.f_max(r, theta) / (1 + np.power(r, 4) - 2 * np.power(r, 2) * np.cos(2 * theta))

    @staticmethod
    def e_total(r, theta):
        return (1 + np.power(r, 2)) / ((1 - np.power(r, 2)) * (1 + np.power(r, 4) - 2 * np.power(r, 2) * np.cos(2 * theta)))

    @staticmethod
    def n_p(r, theta, p):
        p = 1 - p
        e_p = p * AdvancedUtils.e_total(r, theta)
        g_p1 = AdvancedUtils.g_p1(r)
        g_p2_min = AdvancedUtils.g_p2_min(r, theta)
        g_p2_max = AdvancedUtils.g_p2_max(r, theta)
        n_p_min = ((np.log(2 * e_p * np.power(np.sin(theta), 2)) - np.log(g_p1 - g_p2_min)) / (
            np.log(np.power(r, 2)))) - 1
        n_p_max = ((np.log(2 * e_p * np.power(np.sin(theta), 2)) - np.log(g_p1 - g_p2_max)) / (
            np.log(np.power(r, 2)))) - 1
        return n_p_min, n_p_max
    
    @staticmethod
    def get_max_min_el_matrix(p):
        r_resolution = 101
        theta_resolution = 101
        el_matrix = np.zeros([r_resolution-1, theta_resolution-1])
        for r_index in list(range(1, r_resolution)):
            for theta_index in list(range(theta_resolution)):
                r = r_index / r_resolution
                theta = np.pi * theta_index / theta_resolution
                _, el_matrix[r_index-1, theta_index-1] = AdvancedUtils.n_p(r, theta, p)
        return el_matrix

    @staticmethod
    def plot_el_matrix(el_matrix, p, rs, thetas):
        X, Y = np.meshgrid(rs, thetas)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, el_matrix, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('r')
        ax.set_ylabel('W0')
        ax.set_zlabel('Effective Length')
        # ax.set_title('Effective Filter Length for P = ' + str(p*100) + ' %, Min = ' + str(int(np.min(el_matrix))) +
        #              ', Max = ' + str(int(np.max(el_matrix))))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
        ))
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
        ax.view_init(25, -130)
        plt.savefig('Figure 2_b.png', dpi=900)
        # plt.show()

    @staticmethod
    def plot_fig_2(r, p):
        theta_resolution = 1000
        el_array = np.zeros([theta_resolution, 2])
        for theta_index in list(range(theta_resolution)):
            theta = np.pi * theta_index / theta_resolution
            el_array[theta_index, 0], el_array[theta_index, 1] = AdvancedUtils.n_p(r, theta, p)
        x = np.linspace(0, np.pi, theta_resolution)
        plt.plot(x, el_array[:, 0], label='Lower Bound')
        plt.plot(x, el_array[:, 1], label='Upper Bound')
        plt.title(f'r = {r}, P = {p}')
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("Frequency")
        ax.set_ylabel('Effective Filter Length')
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
        ))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 8))
        plt.show()

    @staticmethod
    def plot_mel_scale_poles():
        fs = 16000
        N_filt = 80
        Ps = [0.5, 0.6, 0.7, 0.8, 0.9]
        mel_freqs = SignalUtils.get_mel_freqs(N_filt, fs)
        f1, f2 = SignalUtils.get_f1_f2(mel_freqs, fs)
        # x = np.linspace(0, np.pi, N_filt)
        for p in Ps:
            el_array = np.zeros([N_filt, 2])
            x = np.zeros(N_filt)
            for i in range(N_filt):
                theta = 2 * np.pi * (f1[i] + f2[i]) / (2 * fs)
                sigma = 2 * np.pi * (f2[i] - f1[i]) / (1 * fs)
                pole = np.exp(-sigma + 1j * theta)
                r = abs(pole)
                x[i] = theta
                el_array[i, 0], el_array[i, 1] = AdvancedUtils.n_p(r, theta, p)
            plt.plot(x, el_array[:, 1], label='Upper Bound (P = ' + str(p) + ')')


        # plt.plot(x, el_array[:, 0], label='Lower Bound')

        plt.title(f'Initial Mel-Scale Filters Effective Lengths with different Ps')
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("Frequency")
        ax.set_ylabel('Effective Filter Length')
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
        ))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 8))
        plt.show()

    @staticmethod
    def plot_different_r_mel_scale_poles():
        fs = 16000
        N_filt = 80
        Rs = [0.5, 0.6, 0.7, 0.8, 0.9]
        p = 0.9
        mel_freqs = SignalUtils.get_mel_freqs(N_filt, fs)
        f1, f2 = SignalUtils.get_f1_f2(mel_freqs, fs)
        # x = np.linspace(0, np.pi, N_filt)
        for r in Rs:
            el_array = np.zeros([N_filt, 2])
            x = np.zeros(N_filt)
            for i in range(N_filt):
                theta = 2 * np.pi * (f1[i] + f2[i]) / (2 * fs)
                sigma = 2 * np.pi * (f2[i] - f1[i]) / (1 * fs)
                pole = np.exp(-sigma + 1j * theta)
                x[i] = theta
                el_array[i, 0], el_array[i, 1] = AdvancedUtils.n_p(r, theta, p)
            plt.plot(x, el_array[:, 1], label='Upper Bound (r = ' + str(r) + ')')

        # plt.plot(x, el_array[:, 0], label='Lower Bound')

        plt.title(f'Initial Mel-Scale Filters Effective Lengths with different r for P = {p}')
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("Frequency")
        ax.set_ylabel('Effective Filter Length')
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
        ))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 8))
        plt.show()


    @staticmethod
    def plot_different_r_poles():
        theta_resolution = 100
        Rs = [0.2, 0.4, 0.6, 0.8, 0.99]
        p = 0.9
        for r in Rs:
            el_array = np.zeros([theta_resolution, 2])
            x = np.zeros(theta_resolution)
            for theta_index in range(theta_resolution):
                theta = np.pi * theta_index / theta_resolution
                x[theta_index] = theta
                el_array[theta_index, 0], el_array[theta_index, 1] = AdvancedUtils.n_p(r, theta, p)
            plt.plot(x, el_array[:, 1], label='Upper Bound (r = ' + str(r) + ')')

        # plt.plot(x, el_array[:, 0], label='Lower Bound')

        plt.title(f'Effective Filter Lengths for different r with P = {p}')
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("Frequency (rad)")
        ax.set_ylabel('Effective Filter Length')
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
        ))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 8))
        plt.show()

    @staticmethod
    def plot_different_p_poles():
        theta_resolution = 100
        r = 0.99
        Ps = [0.5, 0.6, 0.7, 0.8, 0.9]
        for p in Ps:
            el_array = np.zeros([theta_resolution, 2])
            x = np.zeros(theta_resolution)
            for theta_index in range(theta_resolution):
                theta = np.pi * theta_index / theta_resolution
                x[theta_index] = theta
                el_array[theta_index, 0], el_array[theta_index, 1] = AdvancedUtils.n_p(r, theta, p)
            plt.plot(x, el_array[:, 1], label='Upper Bound (P = ' + str(p) + ')')

        # plt.plot(x, el_array[:, 0], label='Lower Bound')

        plt.title(f'Effective Filter Lengths for different P with r = {r}')
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("Frequency")
        ax.set_ylabel('Effective Filter Length')
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
        ))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 8))
        plt.show()

    @staticmethod
    def compute_total_energy_matrix(rs, thetas, n_max):
        total_energy_mat = np.zeros([len(rs), len(thetas)])
        for r_index, r in enumerate(rs):
            for theta_index, theta in enumerate(thetas):
                total_energy_mat[r_index, theta_index] = AdvancedUtils.compute_energy(r, theta, n_max)
        return total_energy_mat

    @staticmethod
    def compute_energy(r, theta, length):
        t = torch.linspace(0, length // 2, steps=length // 2)
        t1 = 1 + torch.linspace(0, length // 2, steps=length // 2)
        y_right = torch.pow(r, t) * torch.sin(theta * t1) / torch.sin(torch.tensor(theta))
        y_right_minus = torch.flip(y_right, [0])
        impulse_response = F.conv1d(y_right.view(1, 1, length // 2),
                                    y_right_minus.view(1, 1, length // 2), padding=length // 2)[0, 0, :]
        return torch.sum(torch.pow(impulse_response, 2)).detach().numpy()

    @staticmethod
    def compute_p_effective_length_matrix(total_energy_mat, rs, thetas, p, n_max):
        el_mat = n_max * np.ones([len(rs), len(thetas)])
        for r_index, r in enumerate(rs):
            for theta_index, theta in enumerate(thetas):
                for length in range(2, n_max):
                    energy = AdvancedUtils.compute_energy(r, theta, length)
                    if energy >= p * total_energy_mat[r_index, theta_index]:
                        el_mat[r_index, theta_index] = length
                        break
                print(f'Calculate r = {r} , theta = {theta} : EL = {length}')
        return el_mat





