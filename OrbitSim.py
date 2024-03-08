import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, tan, atan2

'''
This assumes a simplified (2d) orbit model. 
Earth is 0-sized, and located at (0,0). There are no
"inputs".  Just a "free falling" satellite orbit.

Gravity is modeled as G_E/r^2, where G_E is 3.986E14

The code should generate a data file that 
includes the following fields:

* R
* Q
* dt
* meas
* x0
* P0
* truth
'''

def meas(state,R):
    perfect_meas = atan2(state[1],state[0])
    return perfect_meas + la.cholesky(R,lower=True).dot(np.random.randn())

G_E = 3.986E14
# Let's define system parameters first
dt=15 # A measurement occurs every 5 seconds

R = (.2 * pi/180.)**2 # 1/5 degree error
Q = np.diag([.01, .01, .0025, .0025]) # .1m and .05m/s random walk noise
num_steps = 1800 # 10 hours
# Mid-earth orbit is between 2000km and 35000km.
# https://en.wikipedia.org/wiki/Medium_Earth_orbit
# We'll set a mean of 20,000 km, with a standard deviation
# of 3000km
# x will have standard deviation of 1000km
# if at 20,000km, a circular orbit would be 
# v^2 / 20,000,000 = G_E/(20,000,000^2) -> v = 4464.3 (we'll use 5000) m/s
# standard deviation of 800 m/s
x0 = np.array([0, 2E7, 4500, 0])
P0 = np.diag([1E6**2, 3E6**2, 800**2, 800**2])

curr_x = x0 + la.cholesky(P0,lower=True).dot(np.random.randn(4))
true_state = np.zeros((num_steps+1,4))
measurements = np.zeros((num_steps+1))
measurements[0] = meas(curr_x,R)
true_state[0] = curr_x

# Let's now propagate
dt_divider=150
my_dt = dt/dt_divider
my_Q = Q*my_dt

F = np.array([[1,0,my_dt,0],[0,1,0,my_dt],[0,0,1,0],[0,0,0,1]])
for i in range(num_steps):
    comp_Q = np.zeros((4,4))
    for _ in range(dt_divider):
        accel = -G_E *curr_x[:2]/(la.norm(curr_x[:2])**3)
        move_accel = np.concatenate([accel * 0.5*my_dt**2, my_dt*accel])
        curr_x = F.dot(curr_x)+move_accel
        comp_Q = F.dot(comp_Q.dot(F.T))+my_Q
    measurements[i+1] = meas(curr_x,R)
    true_state[i+1]=curr_x + np.random.multivariate_normal(np.zeros(4),comp_Q)
plt.plot(true_state[:,0],true_state[:,1])
ax = plt.gca()
ax.set_aspect('equal')
ax.grid(True)
plt.figure()
# plt.plot(true_state[:,2],true_state[:,3])
# plt.title('Velocity')
# plt.figure()
plt.plot(true_state[:,:2])
plt.legend(['x','y'])
plt.figure()
plt.plot(true_state[:,2:])
plt.legend(['v_x','v_y'])
# plt.plot(true_state[:,2])
# plt.figure()
# plt.plot(true_state[:,3])
plt.show()

np.savez('slide_example2.npz', R=R,Q=Q,
            dt=dt,
            meas=measurements, x0=x0,
            P0=P0, truth=true_state)

np.savez('orbit_run_restricted.npz', R=R,Q=Q,
            dt=dt,
            meas=measurements, x0=x0,
            P0=P0 )            