import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True

# Change these values to get the plots that you need.
prefix = 'slide_example'
ekf_data = np.load('ekf_'+prefix+'_res.npz')
fg_data = np.load('fg_'+prefix+'_res.npz')

truth = ekf_data['truth']
ekf_res = ekf_data['ekf_ref'] # ref is a typo.  should be res=results
fg_res = fg_data['fg_res']


####### Figure 1
plt.figure()
plt.plot(ekf_res[:,0],ekf_res[:,1],c='b', label='estimate')
plt.plot(truth[:,0],truth[:,1],'r--',label='truth')
plt.legend(loc='center left', bbox_to_anchor=(0.5,0.5))
plt.grid(which='major',linestyle='-.', color = '#CCCCCC')

ax=plt.gca()
ax.set_aspect('equal')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Add a box around the plot
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.axhline(y_min, color='gray', linewidth=1.5)
ax.axhline(y_max, color='gray', linewidth=1)
ax.axvline(x_min, color='gray', linewidth=1)
ax.axvline(x_max, color='gray', linewidth=1.5)

formatter=FuncFormatter(lambda x, pos: f"{x/1E6:.0f}")
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter(formatter)

# Set axis labels outside the figure with the spine location in inches
ax.set_xlabel(r'x (m$\times 10^6$)', labelpad=65)  # Adjust labelpad as needed
ax.set_ylabel(r'y (m$\times 10^6$)', labelpad=85)  # Adjust labelpad as needed
yticks = plt.yticks()
print(type(yticks[0]),type(yticks[1]))
print(yticks[0])
yticks[1][1].set_visible(False)
print((yticks[1]))

plt.savefig('ekf_'+prefix+'.pdf',bbox_inches='tight')


######### Figure 2

plt.figure()
plt.plot(fg_res[:,0],fg_res[:,1],c='b', label='estimate')
plt.plot(truth[:,0],truth[:,1],'r--',label='truth')
plt.grid(which='major',linestyle='-.', color='#CCCCCC')
plt.legend(loc='center left', bbox_to_anchor=(0.5,0.5))


ax=plt.gca()
ax.set_aspect('equal')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Add a box around the plot
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.axhline(y_min, color='gray', linewidth=1.5)
ax.axhline(y_max, color='gray', linewidth=1)
ax.axvline(x_min, color='gray', linewidth=1)
ax.axvline(x_max, color='gray', linewidth=1.5)

formatter=FuncFormatter(lambda x, pos: f"{x/1E6:.0f}")
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter(formatter)

# Set axis labels outside the figure with the spine location in inches
ax.set_xlabel(r'x  (m$\times 10^6$)', labelpad=65)  # Adjust labelpad as needed
ax.set_ylabel(r'y (m$\times 10^6$)', labelpad=85)  # Adjust labelpad as needed


# ax.set_xlabel('x (m x 1E6)', labelpad=45)
# ax.set_ylabel('y (m x 1E6)', position=(0,1))

plt.savefig('fg_'+prefix+'.pdf',bbox_inches='tight')
plt.show()


