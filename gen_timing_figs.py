import numpy as np
import matplotlib.pyplot as plt

data = np.load('timing_res_1801_3.npz')

one_step = data["create_and_solve_L"]
full_opt = data["full_opt"]
sizes = data["call_sizes"]

plt.plot(sizes[1:],full_opt[1:],'.',markersize=4)
plt.xlabel('number of time steps',fontsize=14)
plt.ylabel('time (s)', fontsize=14)
plt.savefig('timing_full.pdf',bbox_inches='tight')
plt.show()
