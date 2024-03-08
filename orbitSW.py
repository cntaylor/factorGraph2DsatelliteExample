import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, sqrt, ceil, atan2
from numba import jit
import time
from tqdm import tqdm

def dense_2_sp_lists(M: np.array, tl_row : int, tl_col: int, row_vec=True):
    '''
    This function takes in a dense matrix and turns it into a flat array.
    Corresponding to that array are row and column entries with
    tl_row and tl_col giving the top left of all the entries
    
    Inputs:
        M : The dense to matrix to convert (np.array)
        tl_row: the top row for the matrix (int)
        tl_col: the left-most column for the matrix (int)
        row_vec:  In the corner case where M is 1d, should 
                it be a row or column vector?
        
    Returns:  a tuple with 3 lists (np.array) in it
    '''
    data_list = M.flatten()
    if len(M.shape)==2:
        rows,cols = M.shape
    elif len(M.shape)==1:
        if row_vec:
            rows=1
            cols=len(M)
        else:
            cols=1
            rows=len(M)
    else:
        assert False, 'M must be 1d or 2d!'
    row_list = np.zeros(len(data_list))
    col_list = np.zeros(len(data_list))
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            row_list[idx] = i+tl_row
            col_list[idx] = j+tl_col
    return (data_list,row_list,col_list)

'''
    L's columns will be 3*N big, where N is the number of timesteps
    Rows will be organized organized dynamics first, then measurements:

'''

# To make numba fast, try making these functions outside the class
@jit
def fast_prop_one_timestep(state,T,dt,GE,n_timesteps):
    going_out = state.copy()
    for _ in range(n_timesteps):
        dist = sqrt(going_out[0]*going_out[0] + going_out[1]*going_out[1])
        add_vec = GE * dt/(dist**3) *\
            np.array([going_out[0]/2* dt, 
                        going_out[1]/2* dt,
                        going_out[0], 
                        going_out[1]])
        going_out = T.dot(going_out) - add_vec
    return going_out

@jit
def fast_F_mat(state,T,dt,GE,n_timesteps):
    F = np.eye(4)
    curr_state = state.copy()
    for _ in range(n_timesteps):
        x = curr_state[0]
        y = curr_state[1]
        dist = sqrt(x*x + y*y)
        # dist = la.norm(curr_state[:2])  # Turns out la.norm is really, really slow
        k = dt * GE/(dist**5)

        t_mat = np.array([[-dt*(y**2-2*x**2)/2, 3*x*y* dt/2.0, 0, 0],
                            [3*x*y* dt/2.0, - dt*(x**2-2*y**2)/2, 0, 0],
                            [2*x**2-y**2, 3*x*y, 0, 0],
                            [3*x*y, 2*y**2-x**2, 0, 0]])
        F = (T+ k*t_mat).dot(F)
        #Now update the state as well
        add_vec = GE* dt/(dist**3) *\
            np.array([x/2* dt, 
                        y/2* dt,
                        x, 
                        y])
        curr_state = T.dot(curr_state) - add_vec

    return F

class satelliteSlidingWindow:
    def __init__(self, meas: float, R: np.ndarray, 
                Q : np.ndarray, P0 : np.ndarray, 
                x0: np.ndarray = np.array([0, 2E7, 4500, 0]),
                dt: float = 5, sw_size: int=10):
        
        self.dt = dt
        if self.dt > 1:
            self.prop_dt = self.dt / ceil(self.dt)
            self.n_timesteps = int(ceil(self.dt))
        else:
            self.prop_dt = self.dt
            self.n_timesteps=1
        self.sw_size = sw_size

        self.T = np.array([1,0,self.prop_dt,0, 0,1.,0,self.prop_dt, 0,0,1,0, 0,0,0,1]).reshape((4,4))
        self.GE = 3.986E14

        self.meas = np.array([meas])
        self.S_Q_inv = la.inv(la.cholesky(Q*dt))
        self.S_R_inv = 1/sqrt(R)
        S_P0_inv = la.inv(la.cholesky(P0))
        self.prior_G = S_P0_inv
        self.prior_xbar = x0
        self.states = np.zeros((1,4))
        self.states[0] = x0
        self.opt_call_sizes=[]
        self.opt_call_times=[]
        self.solve_L_times=[]
        self.solve_and_form_L_times=[]

    def N(self):
        return len(self.states)
    
    def prop_one_timestep(self, state):
        return fast_prop_one_timestep(state, self.T, self.prop_dt, self.GE, self.n_timesteps)
        # going_out = state.copy()
        # for _ in range(self.n_timesteps):
        #     dist = sqrt(going_out[0]*going_out[0] + going_out[1]*going_out[1])
        #     add_vec = self.GE*self.prop_dt/(dist**3) *\
        #         np.array([going_out[0]/2*self.prop_dt, 
        #                   going_out[1]/2*self.prop_dt,
        #                   going_out[0], 
        #                   going_out[1]])
        #     going_out = self.T.dot(going_out) - add_vec
        # return going_out

    def state_idx(self, i: int) -> int:
        return 4*i

    def dyn_idx(self, i:int) -> int:
        '''
        This function takes the dynamics index, where this related 
        timestep i-1 to timestep i

        Input:  i-- integer ranging from 1 to N
        Returns a row index (integer)
        '''
        return 4*(i-1)

    def meas_idx(self, i:int) -> int:
        return (self.N()-1)*4 + i
    
    def prior_idx(self) -> int:
        return (self.N()-1)*4 + self.N()
    
    def add_one_timestep(self, meas: float):
        # Predict and store the new state
        next_state = self.prop_one_timestep(self.states[-1])
        self.states = np.vstack((self.states,next_state))
        # store the measurement for the new timestep
        self.meas = np.append(self.meas,meas)
        if self.N() > self.sw_size:
            self.marginalize_state()

    def marginalize_state(self):
        '''
        Remove the oldest state value and marginalize the probability
        to make it a prior
        
        Inputs: None
        Returns: Nothing, just modifies internal state
        '''
        # Run QR decomposition and all that fancy jazz
        ## 1.  Create a row with the things to be marginalized first
        L = self.create_L()
        ## 2a.  Find the rows with non-zero entries
        row_idx,_,_ = sp.find(L[:,:4])
        row_idx = np.unique(row_idx)
        ## 2b.  Create the sub-matrix
        sub_L = L[row_idx,:8]
        ## 3.  Run QR decomposition
        sub_L = sub_L.todense()
        Q,R = la.qr(sub_L, mode='economic')
        ## 3a.  Modify y by the same rotation matrix
        y = self.create_y()
        sub_y = y[row_idx]
        rotated_y = Q.T.dot(sub_y)
        ## 4 & 5.  Eliminate row and column and move remainder into prior_G
        self.prior_G = R[4:,4:]
        self.prior_xbar = self.states[1] + la.pinv(self.prior_G).dot(rotated_y[4:])

        # Shorten the arrays
        self.meas = self.meas[1:]
        self.states = self.states[1:]

    def H_mat(self, state):
        '''
        Take in a current state and return the H matrix (measurement derivative matrix)
        Inputs:  state -- a 3-vector (np.array)
        Returns a n_measurements X 3 matrix (np.array)
        '''
        going_out = np.zeros(4)
        dist_sq = state[0]*state[0] + state[1]*state[1]
        going_out[0] = -state[1]/dist_sq
        going_out[1] = state[0]/dist_sq
        return going_out
    
    def F_mat(self,state):
        '''
        Takes in a current state and finds the derivative of the
        propagate forward one step function (prop_one_timestep)
        
        Inputs: state -- a 4-vector (np.array)
        
        Returns a 4x4 np.array
        '''
        return fast_F_mat(state, self.T, self.prop_dt, self.GE, self.n_timesteps)
        # F = np.eye(4)
        # curr_state = state.copy()
        # for _ in range(self.n_timesteps):
        #     x = curr_state[0]
        #     y = curr_state[1]
        #     dist = sqrt(x*x + y*y)
        #     # dist = la.norm(curr_state[:2])  # Turns out la.norm is really, really slow
        #     k = self.prop_dt*self.GE/(dist**5)

        #     t_mat = np.array([[-self.prop_dt*(y**2-2*x**2)/2, 3*x*y*self.prop_dt/2.0, 0, 0],
        #                       [3*x*y*self.prop_dt/2.0, -self.prop_dt*(x**2-2*y**2)/2, 0, 0],
        #                       [2*x**2-y**2, 3*x*y, 0, 0],
        #                       [3*x*y, 2*y**2-x**2, 0, 0]])
        #     F = (self.T+ k*t_mat).dot(F)
        #     #Now update the state as well
        #     add_vec = self.GE*self.prop_dt/(dist**3) *\
        #         np.array([x/2*self.prop_dt, 
        #                   y/2*self.prop_dt,
        #                   x, 
        #                   y])
        #     curr_state = self.T.dot(curr_state) - add_vec

        # return F

    def create_L(self):
        H_size = 4
        F_size=16 # Should be state size**2
        prior_size = 16
        nnz_entries = 2*F_size*(self.N()-1) + H_size*self.N() + prior_size
        data_l = np.zeros(nnz_entries)
        row_l = np.zeros(nnz_entries,dtype=int)
        col_l = np.zeros(nnz_entries,dtype=int)
        t_e = 0 #total number of entries so far
        # Put all the dynamics entries into L
        for i in range(1,self.N()):
            mat1 = self.S_Q_inv.dot(self.F_mat(self.states[i-1]))
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat1,self.dyn_idx(i),self.state_idx(i-1))
            t_e +=F_size
            
            mat2 = -self.S_Q_inv
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat2,self.dyn_idx(i),self.state_idx(i))
            t_e +=F_size
        
        # Now do the measurements
        for i in range(self.N()):
            # for S_R_inv a scalar
            mat = self.S_R_inv*self.H_mat(self.states[i])
            data_l[t_e:t_e+H_size], row_l[t_e:t_e+H_size], col_l[t_e:t_e+H_size] = \
                dense_2_sp_lists(mat,self.meas_idx(i),self.state_idx(i))
            t_e += H_size

        # Now put in the prior
        data_l[t_e:t_e+prior_size], row_l[t_e:t_e+prior_size], col_l[t_e:t_e+prior_size] = \
            dense_2_sp_lists(self.prior_G,self.prior_idx(),self.state_idx(0))
        t_e += prior_size


        return sp.csr_matrix((data_l,(row_l,col_l)))

    def create_y(self, state_vec=None):
        if state_vec is not None:
            state_data = self.vec_to_data(state_vec)
        else:
            state_data = self.states
        y = np.zeros(4*(self.N()-1)+self.N()+4) # last 4 for prior
        for i in range(1,self.N()):
            # predicted measurement for dynamics is f(x_{k-1})-x_k
            pred_meas = self.prop_one_timestep(state_data[i-1])-state_data[i]
            y[self.dyn_idx(i):self.dyn_idx(i+1)]=self.S_Q_inv.dot(-pred_meas)
        # Now do the measurements received
        for i in range(self.N()):
            pred_meas = atan2(state_data[i,1],state_data[i,0])
            tmp= self.meas[i]-pred_meas
            if tmp > pi:
                tmp -= 2*pi
            if tmp < -pi:
                tmp += 2*pi
            y[self.meas_idx(i):self.meas_idx(i+1)] = self.S_R_inv * tmp
        # Now do the prior
        y[self.prior_idx():self.prior_idx()+4] = self.prior_G.dot(self.prior_xbar - state_data[0])
        return y

    def vec_to_data(self,vec):
        going_out = np.zeros((self.N(),4))
        for i in range(self.N()):
            going_out[i] = vec[i*4:(i+1)*4]
        return going_out

    def add_delta(self,delta_x: np.array = None) -> np.array:
        '''
        This takes the current state and adds on delta_x.
        It DOES NOT  modify any internal state

        Inputs: delta_x, a self.N() X 3 vector (np.array)
        Returns: a full state vector of the same size
        '''
        going_out = np.zeros(self.N()*4)
        if delta_x is None:
            delta_x = np.zeros(self.N()*4)
        for i in range(self.N()):
            going_out[i*4:(i+1)*4] = self.states[i]+ \
                delta_x[i*4:(i+1)*4]

        return going_out

    def update_state(self,delta_x):
        '''
        Changes the internal states data structure with the
        delta_x that is included here
        
        Returns nothing (only modifies internal state)
        '''
        for i in range(self.N()):
            self.states[i] += delta_x[i*4:(i+1)*4]       

    def opt(self, max_iters:int = 100):
        '''
        Create the Jacobian matrix (L) and the residual vector (y) for
        the current state.  Find the best linear approximation to minimize y
        and move in that direction.  Repeat until convergence. 
        (This is a Gauss-Newton optimization procedure)
        '''
        tic1 = time.perf_counter()
        finished=False
        num_iters=0
        while not finished:
            tic3 = time.perf_counter()
            L= self.create_L()
            # print ('L shape is',L.shape)
            # plt.spy(L,markersize=1)
            # plt.show()
            # for i in range(9):
            #     print('For column',i)
            #     test_Jacobian(self,i,.0001)
            # test_Jacobian(self,2)
            y= self.create_y()
            tic2=time.perf_counter()
            M = L.T.dot(L)
            Lty = L.T.dot(y)
            delta_x = spla.spsolve(M,Lty)
            toc2=time.perf_counter()
            scale = 1
            scale_good=False
            # A measure of how much
            #improvement you actually expect from this step
            pred_delta_y_norm=la.norm(L.dot(delta_x))
            ratio2 = pred_delta_y_norm/la.norm(y)
            # print('ratio2 is ',ratio2,'delta y norm is ',pred_delta_y_norm, 'delta x norm is ',la.norm(delta_x))
            if ratio2<1E-4 or pred_delta_y_norm<1E-6:
                finished=True
                print('y ratio is too small to run iteration',num_iters,'ratio is:',ratio2)
            else:
                while not scale_good:
                    next_y = self.create_y(self.add_delta(delta_x*scale))
                    pred_y = y-L.dot(delta_x*scale)
                    y_mag = y.T.dot(y)
                    ratio = (y_mag - next_y.dot(next_y))/(y_mag-pred_y.dot(pred_y))
                    if ratio < 4. and ratio > .25:
                        scale_good = True
                    else:
                        scale /= 2.0
                        if scale < .1:
                            print('Your derivatives are probably wrong!  scale is',scale,'ratio is',ratio, 'ratio2 is',ratio2)
                    assert(scale > 1E-6)
                num_iters+=1
                self.update_state(delta_x*scale)
                # print('iteration',num_iters,'delta_x length was',la.norm(delta_x*scale), 'scale was',scale)
                finished = la.norm(delta_x)<1 or num_iters >= max_iters
        toc1=time.perf_counter()
        self.opt_call_sizes.append(self.N())
        self.opt_call_times.append(toc1-tic1)
        self.solve_L_times.append(toc2-tic2)
        self.solve_and_form_L_times.append(toc2-tic3)

        
def test_Jacobian(batch_uni, col, dx = .01):
    curr_y = batch_uni.create_y()
    orig_state = batch_uni.add_delta()
    delta_x = np.zeros(orig_state.shape)
    delta_x[col] = dx
    next_state = batch_uni.add_delta(delta_x)
    next_y = batch_uni.create_y(next_state)
    num_J = (next_y-curr_y)/dx
    J = batch_uni.create_L()[:,col].toarray()
    J = J.reshape(num_J.shape)    
    diff = num_J+J
    print(diff)
    print(np.max(diff))
    print(np.min(diff))

    
def RMSEs(est_states: np.array, true_states: np.array) -> float:
    '''
    This function takes the estimated states and compares them 
    against the true states.  The estimated states do _not_ need to
    be as long as the true_states.  It returns 3 values:
    * the total RMSE
    * the RMSE of just the positions
    * the RMSE of just the velocities

    Inputs:
        est_states: a Nx4 np.array with a different state value in each row
        true_states: an Mx4 (where M>=N) np.array
    
    Returns: tuple of three values (see description about)
    '''
    assert len(true_states)>= len(est_states), 'Not enough entries in true_states'
    total_RMSE = sqrt(np.sum(np.square(true_states[:len(est_states)]-est_states))/(len(est_states)*4))
    pos_RMSE = sqrt(np.sum(np.square(true_states[:len(est_states),:2]-est_states[:,:2]))/(len(est_states)*2))
    vel_RMSE = sqrt(np.sum(np.square(true_states[:len(est_states),2:]-est_states[:,2:]))/(len(est_states)*2))
    return (total_RMSE, pos_RMSE, vel_RMSE)

def errors(est_data, truth) -> np.array:
    return 
def ANEES():
    #TODO:  implement this function to be useful
    pass

if __name__ == '__main__':
    prefix = 'slide_example2'
    data = np.load(f'{prefix}.npz')
    meas = data['meas']
    truth = data['truth']
    dt = data['dt']
    R = data['R']
    Q = data['Q']
    P0 = data['P0']
    x0 = data['x0']
    plt.plot(truth[:,0],truth[:,1],'r',label='truth')
    
    data_len = len(meas)
    
    window_size= 2 #data_len #int(data_len/200)*100
    # Can either store the oldest things (most optimized)
    # or always store the most recent
    # True = the oldest ones
    store_most_opt = False

    #max iterations each timestep
    max_iters=1

    opt_class = satelliteSlidingWindow(meas[0], R, Q, P0, dt=dt, sw_size=window_size)
    # print (" a sample F")
    # print(opt_class.F_mat(opt_class.states[10]))
    opt_states = np.zeros((data_len,4)) #back sliding window
    rt_states = np.zeros((data_len,4)) # real time type states
    opt_states[0] = x0 # In case not over-written later .. basically a special case when window_size=1
    rt_states[0] = x0
    for i in tqdm(range(1,data_len)):
        opt_class.add_one_timestep(meas[i])
        # print('Optimized for index',i)
        
        opt_class.opt(max_iters)
        if i> (window_size-1):
            opt_states[i-window_size+1] = opt_class.states[0]
        rt_states[i]=opt_class.states[-1]
    opt_states[-window_size:]= opt_class.states
    # rt_states[-window_size:]= opt_class.states
    
    print ('RMSEs',RMSEs(opt_states,truth))
    errors = opt_states-truth[:len(opt_states)]

    plt.plot(opt_states[:,0],opt_states[:,1],'m',label='opt_final')


    plt.legend()
    # plt.savefig(f'{prefix}_SW_{window_size}_res.png')
    
    fig,axs= plt.subplots(2,1)
    axs[0].set_title('position error')
    axs[0].plot(errors[:,0],label='x')
    axs[0].plot(errors[:,1],label='y')
    axs[0].legend()
    axs[1].set_title('velocity error')
    axs[1].plot(errors[:,2],label='x')
    axs[1].plot(errors[:,3],label='y')
    
    plt.figure()
    ekf_res = np.load('ekf_res.npy')
    plt.plot(ekf_res[:,0], ekf_res[:,1], c='c', label='EKF')
    plt.plot(opt_states[:,0],opt_states[:,1],c='b', label=f'delay={window_size}')
    plt.plot(rt_states[:,0], rt_states[:,1], 'g', label='real-time')
    plt.plot(truth[:,0],truth[:,1],'r--',label='truth')
    plt.legend()
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.savefig(f'SW_{prefix}_{window_size}.png')

    plt.figure()
    ekf_err=np.sqrt(np.sum(np.square(ekf_res[:data_len,:2]-truth[:data_len,:2]),axis=1))
    rt_sw_err=np.sqrt(np.sum(np.square(rt_states[:,:2]-truth[:data_len,:2]),axis=1))
    delayed_sw_err=np.sqrt(np.sum(np.square(opt_states[:,:2]-truth[:data_len,:2]),axis=1))
    plt.plot(ekf_err,c='c',label='EKF')
    plt.plot(delayed_sw_err,c='b',label=f'delay={window_size}')
    plt.plot(rt_sw_err,c='g',label='real-time')
    plt.legend()
    plt.savefig(f'SW_errs_{prefix}_{window_size}.png')
    with open(f'err_res_windows.txt', 'a') as f:
        f.write('\n--------------------------------------\n')
        f.write (f'For window size {window_size}\n')    
        f.write(f'EKF RMSE error: {sqrt(np.sum(np.square(ekf_err))/data_len)}\n')
        f.write(f'RT SW RMSE error: {sqrt(np.sum(np.square(rt_sw_err))/data_len)}\n')
        f.write(f'Delayed SW RMSE error: {sqrt(np.sum(np.square(delayed_sw_err))/data_len)}\n')
        f.write(f'EKF RMSE error -- last half: {sqrt(2*np.sum(np.square(ekf_err[int(data_len/2):]))/data_len)}\n')
        f.write(f'RT SW RMSE error -- last half: {sqrt(2*np.sum(np.square(rt_sw_err[int(data_len/2):]))/data_len)}\n')
        f.write(f'Delayed SW RMSE error -- last half: {sqrt(2*np.sum(np.square(delayed_sw_err[int(data_len/2):]))/data_len)}\n')

    np.savez(f'timing_res_{window_size}.npz', call_sizes=opt_class.opt_call_sizes, full_opt = opt_class.opt_call_times,
             solve_L_times = opt_class.solve_L_times, create_and_solve_L = opt_class.solve_and_form_L_times)
    plt.figure()
    plt.plot(opt_class.opt_call_sizes,opt_class.opt_call_times,'*')
    plt.title('Time vs size, full optimization')
    plt.savefig(f'Opt_timing_{window_size}.png')
    
    plt.figure()
    plt.plot(opt_class.opt_call_sizes,opt_class.solve_L_times,'*')
    plt.title('Time vs size, Solve L')
    plt.savefig(f'LA_timing_{window_size}.png')

    plt.figure()
    plt.plot(opt_class.opt_call_sizes,opt_class.solve_and_form_L_times,'*')
    plt.title('Time vs size, Create and Solve L')
    plt.savefig(f'LA_create_timing_{window_size}.png')
    plt.show()



# ######### Test if dense_2_sp_lists is working   ##########
# # Create a block diagonal matrix, plus two matrices off the diagonal
# matrix1 = np.array([0,1.,1.,0]).reshape((2,2))
# big_data_l = np.zeros(7*4)
# big_row_l = np.zeros(7*4,dtype=int)
# big_col_l = np.zeros(7*4,dtype=int)
# # The block diagonal matrices
# for i in range(5):
#     big_data_l[i*4:(i+1)*4],big_row_l[i*4:(i+1)*4],big_col_l[i*4:(i+1)*4] = \
#         dense_2_sp_lists(matrix1,i*2,i*2)
# # A matrix in the top right
# i=5
# matrix2= np.eye(2)
# big_data_l[i*4:(i+1)*4],big_row_l[i*4:(i+1)*4],big_col_l[i*4:(i+1)*4] = \
#     dense_2_sp_lists(matrix2,0,8)
# # A matrix at the bottom left
# i=6
# big_data_l[i*4:(i+1)*4],big_row_l[i*4:(i+1)*4],big_col_l[i*4:(i+1)*4] = \
#     dense_2_sp_lists(matrix2,8,0)

# L = sp.csr_matrix((big_data_l,(big_row_l,big_col_l)))

# plt.spy(L)
# plt.show()

