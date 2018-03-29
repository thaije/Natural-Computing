import numpy as np
from matplotlib import pyplot as plt

err = np.arange(.01,1.01,.01)
beta = .5*np.log((1-err)/err)

plt.plot(err,beta)
plt.title('Weight for different errors')
plt.xlabel('Error')
plt.ylabel('Beta')
plt.show()