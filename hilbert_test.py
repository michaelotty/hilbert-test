#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

t_total = 1.0
fs = 10e3

n = int(t_total * fs)

t = np.linspace(0.0, t_total, n)
y = np.sin(2 * np.pi * 2000.0 * t)

h = hilbert(y)

yy = np.fft.fft(h, norm='ortho')
f = np.fft.fftfreq(yy.size, d=1/fs)

plt.subplot(311)
plt.plot(f, np.abs(yy))
plt.subplot(312)
plt.plot(t, np.abs(h))
plt.plot(t, np.real(h))
plt.plot(t, np.imag(h))
plt.subplot(313)
plt.plot(t, np.unwrap(np.angle(h)))

plt.show()

