from matplotlib import pyplot as plt
import numpy as np


"""
Contains the functions for plotting modulations
"""

# TODO
#  Make plots only use a sensible number of samples of the data
#  Animate plots
#  Add FFt bins


def plot(data, **kwargs):
    if kwargs['type'] == "specgram":
        plt.specgram(data, NFFT=kwargs['nfft'], Fs=kwargs['fs'])
        plt.title(kwargs['title'])
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")

    elif kwargs['type'] == 'psd':
        plt.psd(data, NFFT=kwargs['nfft'], Fs=kwargs['fs'])
        plt.title(kwargs['title'])
        plt.axhline(0, color='lightgray')  # x = 0
        plt.axvline(0, color='lightgray')  # y = 0
        plt.grid(True)

    elif kwargs['type'] == 'iq':
        plt.scatter(data.real, data.imag)
        plt.ylabel("Imaginary")
        plt.xlabel("Real")
        plt.title(kwargs['title'])
        plt.axhline(0, color='lightgray')  # x = 0
        plt.axvline(0, color='lightgray')  # y = 0

        # Figure out the axis sizes
        ax_max = round(np.max(np.abs(data))) + 0.2

        plt.xlim(-1*ax_max, ax_max)
        plt.ylim(-1*ax_max, ax_max)

    elif kwargs['type'] == "fft":
        if 'nfft' in kwargs.keys():
            nfft = kwargs['nfft']
        else:
            nfft = 1024

        S = np.fft.fftshift(np.abs(np.fft.fft(data)))
        S_mag = np.abs(S)
        f_axis = np.arange(kwargs['fs'] / -2, kwargs['fs'] / 2, kwargs['fs'] / len(data))
        if len(f_axis) > len(S_mag):
            f_axis = f_axis[0:len(S_mag)]

        plt.plot(f_axis, S_mag)
        plt.title(kwargs['title'])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")

    elif kwargs['type'] == "time":
        t = kwargs['t']
        n = kwargs['n']

        if n == 0:
            n = len(data)
        elif n > len(data):
            n = len(data)

        plt.plot(t[0:n], np.real(data[0:n]))
        plt.plot(t[0:n], np.imag(data[0:n]))
        plt.title(kwargs['title'])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    elif kwargs['type'] == "view":
        env = None
        if kwargs['subtype'] == "phase":
            env = np.angle(data)

        elif kwargs['subtype'] == 'amp':
            plt.title('Amplitude View')
            plt.xlabel("Samples (s)")
            plt.ylabel("Power")
            env = np.abs(data)

        elif kwargs['subtype'] == 'freq':
            phase = np.unwrap(np.angle(data))
            env = np.diff(phase) / (2*np.pi) * kwargs['fs']

        plt.plot(env)

    plt.show()




