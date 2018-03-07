import numpy as np
import matplotlib.pyplot as plt

def fit(x):
    return abs(x)


w = np.linspace(0.1,0.90,9)
r = 1

maxiter = 50

for wi in range(len(w)):
    x = np.zeros(maxiter+1)
    fit_x = np.zeros(maxiter+1)
    v = np.zeros(maxiter+1)

    x_0 = np.random.uniform(-10,10)
    v_0 = np.random.uniform(-10,10)

    x[0] = x_0
    fit_x[0] = fit(x[0])
    v[0] = v_0

    for i in range(1,maxiter+1):
        best_x = np.min(fit_x[0:i])
        best_xi = np.argmin(x[0:i])
        best_x_v = v[best_xi]

        v[i] = w[wi] * v[i-1] + r * (best_x - x[i-1])

        x[i] = x[i-1] + v[i]
        fit_x[i] = fit(x[i])


    plt.plot(x)

plt.xlabel("Step t")
plt.ylabel("x")
labels = ["w = " + str(w[i]) for i in range(len(w))]
plt.legend(labels)
plt.tight_layout()