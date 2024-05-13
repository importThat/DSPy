import numpy as np
import time

# Need to test the speeds of the two wave functions


def WaveTrig(f, t, theta, A):
    z = A * np.cos(2 * np.pi * f * t + theta) + A * 1j * np.sin(2 * np.pi * f * t + theta)
    return z


def WaveExp(f, t, theta, A):
    z = A * np.exp(1j * (2 * np.pi * f * t + theta))
    return z

t = 1/20000000 * np.arange(1 * 20000000)
f = np.random.randint(100, 200, t.shape)
theta = np.random.randint(0, 2, t.shape)
A = np.random.random(t.shape)

start = time.time()
z = WaveTrig(f, t, theta, A)
trig_end = time.time()

trig_dur = trig_end - start

del z

start = time.time()
z = WaveExp(f, t, theta, A)
exp_end = time.time()

exp_dur = exp_end - start

print(f"Trig equation took {trig_dur} seconds\nExponential equation took {exp_dur} seconds")

# Looks like the trig equation is faster! I'm surprised by that
