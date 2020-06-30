import numpy as np
import matplotlib.pyplot as plt
from utils.common_utils import set_default_plot_params, save_fig

set_default_plot_params

plt.figure()

gammaDivGamma_e = np.linspace(0.1, 0.99, num=1000)
# gammaEval = 0.99
# S = 10
# A = 2
# n_params = S * A
# coeff = (gammaEval - gamma ) / (2 * gamma)
coeff = 0.5 * (1 / gammaDivGamma_e - 1)
# coeff /= n_params
plt.plot(gammaDivGamma_e, (coeff))


plt.grid(True)
plt.xlabel(r'$\gamma / \gamma_e$')
plt.ylabel(r'$\lambda = \frac{\gamma_e - \gamma}{2\gamma}$')
# plt.legend()
#
save_PDF = True  # False \ True
if save_PDF:
	save_fig('reg_coeff')
else:
	plt.title('Activation Reg. Coeff.')

plt.show()

print('done')
