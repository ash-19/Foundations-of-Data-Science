import matplotlib.pyplot as plt
from scipy.stats import laplaceimport numpy as npx = np.linspace(-3, 3, 100)plt.plot(x, laplace.pdf(x),linewidth=2.0, label="laplace PDF")plt.plot(x, laplace.cdf(x),linewidth=2.0, label="laplace CDF")plt.legend(bbox_to_anchor=(.35,1))
plt.show()