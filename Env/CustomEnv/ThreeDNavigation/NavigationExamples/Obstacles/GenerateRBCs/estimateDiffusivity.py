import math
import numpy as np
import matplotlib.pyplot as plt
import copy

Dr = 0.161
Dt = 2.145e-13
tau = 1/Dr # tau about 6.211180124223603
a = 1e-6
T = 0.1*tau
v= 2*a/T
xyStd = math.sqrt(2*T*Dt) / a
kb = 1.38e-23
T = 293.15
mu = 0.89e-3

# for blood https://wiki.anton-paar.com/en/whole-blood/
T = 273 + 37
mu = 2.78e-3
a = 0.5e-6

D0 = kb * T / 6.0 / np.pi/mu/a

Dr0 = kb * T / 8.0 / np.pi / mu / a**3


tau = 1/Dr0 # tau about 2.04 s

T = 0.02*tau
v= 2*a/T
xyStd = math.sqrt(2*T*D0) / a
angleStd = math.sqrt(2*T*Dr0) * 180 / np.pi
print(Dt, D0, Dr, Dr0, tau)
print(xyStd, angleStd)