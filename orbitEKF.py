import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, tan, atan2, floor

def plot_1d_est_with_cov(est, covs, truth=None, xs=None, sigmas=3):
    '''
    This function takes in (n,) numpy arrays and plots the estimate with
    "error bars" around the estimate.  Can also plot the truth.  The estimate
    will be red, the 
    
    Args:
        est:  The estimate values over time. An (n,) array
        covs:  The covariance over time.  An (n,) array
        truth:  (optional)  The true values over time.  An (n,) array
        xs: (optional) If nothing passed in, will plot 1->n on the x axis.
            Otherwise, will put xs along the x axis
        sigmas:  (default=3) How many sigmas out to put the "error bars"
            when plotting

    Returns:  nothing. Just plots stuff using matplotlib
    '''
    assert len(est)==len(covs), 'est and covs must be the same length'
    if truth is not None:
        assert len(est)==len(truth), 'est and truth must be the same length'
    if xs is not None:
        assert len(est)==len(xs), 'est and xs must be the same length'    
    else:
        xs = np.arange(len(est))
    plt.plot(xs,est,c='r', label='estimate')
    top_vals = est + np.sqrt(covs)*sigmas
    low_vals = est - np.sqrt(covs)*sigmas
    plt.fill_between(xs,low_vals,top_vals, facecolor='b',alpha=.5)
    if truth is not None:
        plt.plot(xs,truth,c='k',linestyle='--',label='truth')
        plt.legend()
    
def plot_2d_est_with_cov(est,covs,truth=None, sigmas=3, sigma_decimate=10):
    '''
    This function takes in (n,2) numpy arrays and plots the estimate with
    "error bars" around the estimate.  Can also plot the truth.  The estimate
    will be red, the truth black, and ellipses blue
    
    Args:
        est:  The estimate values over time. An (n,2) array
        covs:  The covariance over time.  An (n,2,2) array
        truth:  (optional)  The true values over time.  An (n,2) array
        sigmas:  (default=3) How many sigmas out to put the "error bars"
            when plotting
        sigma_decimate: (default=10) How many of the values to plot covariances
            around (plus stars on the corresponding locations).  If 1, will plot
            an ellipse around every point

    Returns:  nothing. Just plots stuff using matplotlib
    '''
    assert len(est)==len(covs), 'est and covs must be the same length'
    if truth is not None:
        assert len(est)==len(truth), 'est and truth must be the same length'
    plt.plot(est[:,0],est[:,1],c='b', label='estimate')
    plt.plot(est[0::sigma_decimate,0], est[0::sigma_decimate,1], 'b+')
    
    #Create a circle for plotting
    angs = np.arange(0,2*pi+.1,.1)
    circ_pts =np.zeros((2,len(angs)))
    circ_pts[0]=np.sin(angs)
    circ_pts[1]=np.cos(angs)
    circ_pts *= sigmas
    for i in range(len(est)):
        if i%sigma_decimate == 0:
            S_cov = la.cholesky(covs[i,:2,:2],lower=True)
            ellipse = S_cov.dot(circ_pts) + est[i,:2].reshape((2,1)) #reshape enables broadcast
            plt.plot(ellipse[0],ellipse[1],'b')
    if truth is not None:
        plt.plot(truth[:,0],truth[:,1],c='r',label='truth')
        plt.plot(truth[0::sigma_decimate,0], truth[0::sigma_decimate,1], 'r+')
        plt.legend()
    ax=plt.gca()
    ax.set_aspect('equal')

prefix='slide_example'

data = np.load(prefix+'.npz')

meas = data['meas']
num_steps = len(meas)-1 #1000
R = data['R']
Q = data['Q']
dt = data['dt'].item()
curr_x = data['x0']
curr_P = data['P0']
truth = data['truth']


est_state = np.zeros((num_steps+1,4))
est_cov = np.zeros((num_steps+1,4,4))

est_state[0] = curr_x
est_cov[0] = curr_P
#See if this fixes things.
G_E = 3.986E14

def f(x,dt):
    accel = -G_E*x[:2]/la.norm(x[:2])**3
    F = np.eye(4)
    F[:2,2:] = np.eye(2) * dt
    accel_add = np.concatenate((dt**2/2 * accel, accel*dt))
    return F.dot(x) + accel_add

def f2(x,dt):
    dt_divider=50
    my_dt = dt/dt_divider
    F = np.eye(4)
    F[:2,2:] = np.eye(2) * my_dt
    for _ in range(dt_divider):
        accel = -G_E *x[:2]/la.norm(x[:2])**3
        move_accel = np.concatenate((accel * 0.5*my_dt**2, my_dt*accel))
        x = F.dot(x)+move_accel
    return x

def h(x):
    return atan2(x[1],x[0])

T = np.array([1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1]).reshape(4,4)
for i in range(num_steps):
    ##Propagate
    # Compute F
    dist = la.norm(curr_x[:2])
    xy= curr_x[0] * curr_x[1]
    x_sq = curr_x[0] * curr_x[0]
    y_sq = curr_x[1] * curr_x[1]
    F = T + (G_E * dt)/(dist**5)* \
        np.array([dt*(2*x_sq-y_sq)/2, 3*xy*dt/2, 0, 0,
            3*xy*dt/2, dt*(2*y_sq-x_sq)/2, 0,0,
            2*x_sq-y_sq, 3*xy, 0, 0,
            3*xy, 2*y_sq-x_sq, 0, 0 ]).reshape(4,4)
        
        # np.array([-dt*x_sq/2, dt*xy/2, 0, 0,
        #     dt*xy/2, -dt*x_sq/2, 0,0,
        #     -y_sq, xy, 0, 0,
        #     xy, -x_sq, 0, 0 ]).reshape(4,4)
    # Propagate x
    curr_x = f(curr_x,dt)
    # Propagate P
    curr_P = F.dot(curr_P.dot(F.T)) + Q*dt

    ## Update
    #Compute H
    dist_sq = curr_x[0]*curr_x[0] + curr_x[1]*curr_x[1]
    H = np.array([-curr_x[1]/dist_sq, curr_x[0]/dist_sq, 0,0])
    #Compute K
    HPH = H.dot(curr_P.dot(H.T))
    K = curr_P.dot(H.T)/(HPH + R)
    # update x
    innov = meas[i+1]-h(curr_x)
    if innov > pi:
        innov -= 2*pi
    if innov < -pi:
        innov += 2*pi
    curr_x += K.dot(innov)
    # update P
    curr_P = (np.eye(4)-np.outer(K,H)).dot(curr_P)

    #Store stuff for plots and things
    est_state[i+1] = curr_x
    est_cov[i+1] = curr_P
    # u,s,v = la.svd(curr_P)
    # print('s is',s)

decimate = int(floor(num_steps/20))
plot_2d_est_with_cov(est_state, est_cov, truth[:(num_steps+1)], sigma_decimate=decimate)
plt.savefig(prefix+'_loc.png')
plt.figure()
plot_1d_est_with_cov(est_state[:,2],est_cov[:,2,2],truth[:(num_steps+1),2])
plt.title('X Velocity')
plt.savefig(prefix+'_x_vel.png')
plt.figure()
plot_1d_est_with_cov(est_state[:,3],est_cov[:,3,3],truth[:(num_steps+1),3])
plt.title('Y Velocity')
plt.savefig(prefix+'_y_vel.png')

plt.figure()
plt.plot(est_state-truth)
plt.legend (['x','y','vx','vy'])
plt.title('errors')
plt.savefig(f'{prefix}_EKF_errors.png')

plt.figure()
plt.plot(est_state[:,0],est_state[:,1],c='b', label='estimate')
plt.plot(truth[:,0],truth[:,1],'r--',label='truth')
plt.legend()
ax=plt.gca()
ax.set_aspect('equal')
plt.savefig('EKF_'+prefix+'.png')
plt.show()

np.savez('ekf_'+prefix+'_res',ekf_res=est_state, truth=truth)