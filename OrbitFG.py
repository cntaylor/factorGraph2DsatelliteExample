import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, sqrt, ceil, atan2


'''
To create a sparse matrix, you have to create three "parallel" arrays, 
one with row indicies, one with column indicies, and one with the actual
values to put in the matrix.  This is a helper function to make this process easier.
'''
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
class satelliteModelBatch:
    '''
    A class to perform the factor graph optimization.  It stores
    all the measurements in the class, has several "helper" functions, and the
    "opt" function that does the actual optimization using the helper functions
    '''
    
    def __init__(self, meas: np.array, R, Q, 
                dt: float = 5, 
                x0:np.array = np.array([0, 2E7, 4500, 0])):
        '''
        Get the measurements, dt (for running the dynamics), 
        the R & Q matrices, and the initial location and initialize
        the hidden variables
        '''
        self.N = len(meas)
        # self.N = 10
        self.dt = dt
        # To do more accurate dynamics, if dt is large, use multiple, smaller
        # timesteps instead
        if self.dt > 1:
            self.prop_dt = self.dt / ceil(self.dt)
            self.n_timesteps = int(ceil(self.dt))
        else:
            self.prop_dt = self.dt
            self.n_timesteps=1
        self.T = np.array([1,0,self.prop_dt,0, 0,1.,0,self.prop_dt, 0,0,1,0, 0,0,0,1]).reshape((4,4))
        self.GE = 3.986E14

        self.meas = meas
        self.S_Q_inv = la.inv(la.cholesky(Q))
        self.S_R_inv = 1/sqrt(R)
        self.states = np.zeros((self.N,4))
        self.states[0] = x0
        self.create_init_state()

    def create_init_state(self):
        '''
        Take x0 and propagate it forward through all timesteps
        '''
        for i in range(1,self.N):
            self.states[i] = self.prop_one_timestep(self.states[i-1])
        
    def prop_one_timestep(self, state):
        '''
        This runs the actual dynamics, i.e. given a value for x (really x_k) find 
        what x should be (by dynamics) at the next timestep (self.dt forward in time) -- x_{k+1}
        '''
        going_out = state.copy()
        for _ in range(self.n_timesteps):
            dist = la.norm(going_out[:2])
            add_vec = self.GE*self.prop_dt/(dist**3) *\
                np.array([going_out[0]/2*self.prop_dt, 
                          going_out[1]/2*self.prop_dt,
                          going_out[0], 
                          going_out[1]])
            going_out = self.T.dot(going_out) - add_vec
        return going_out


    '''
    Helper functions to tell what rows you are indexing into in the 'L' matrix
    '''
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
        return (self.N-1)*4 + i
    
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
        F = np.eye(4)
        curr_state = state.copy()
        for _ in range(self.n_timesteps):
            x = curr_state[0]
            y = curr_state[1]
            dist = la.norm(curr_state[:2])
            k = self.prop_dt*self.GE/(dist**5)

            t_mat = np.array([[-self.prop_dt*(y**2-2*x**2)/2, 3*x*y*self.prop_dt/2.0, 0, 0],
                              [3*x*y*self.prop_dt/2.0, -self.prop_dt*(x**2-2*y**2)/2, 0, 0],
                              [2*x**2-y**2, 3*x*y, 0, 0],
                              [3*x*y, 2*y**2-x**2, 0, 0]])
            F = (self.T+ k*t_mat).dot(F)
            #Now update the state as well
            add_vec = self.GE*self.prop_dt/(dist**3) *\
                np.array([curr_state[0]/2*self.prop_dt, 
                          curr_state[1]/2*self.prop_dt,
                          curr_state[0], 
                          curr_state[1]])
            curr_state = self.T.dot(curr_state) - add_vec

        return F

    def create_L(self):
        '''
        This creates the big matrix (L) given the current state of the whole system
        '''
        # First, determine how many non zero entries (nnz_entries) will be in the
        # sparse matrix. Then create the 3 parallel arrays that will be used to
        # form this matrix
        H_size = 4
        F_size=16 # Should be state size**2
        nnz_entries = 2*F_size*(self.N-1) + H_size*self.N
        data_l = np.zeros(nnz_entries)
        row_l = np.zeros(nnz_entries,dtype=int)
        col_l = np.zeros(nnz_entries,dtype=int)
        t_e = 0 #total number of entries so far
        # Put all the dynamics entries into L
        for i in range(1,self.N):
            mat1 = self.S_Q_inv.dot(self.F_mat(self.states[i-1]))
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat1,self.dyn_idx(i),self.state_idx(i-1))
            mat2 = -self.S_Q_inv
            t_e +=F_size
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat2,self.dyn_idx(i),self.state_idx(i))
            t_e +=F_size
        
        # Now do the measurements
        for i in range(self.N):
            # for S_R_inv a scalar
            mat = self.S_R_inv*self.H_mat(self.states[i])
            data_l[t_e:t_e+H_size], row_l[t_e:t_e+H_size], col_l[t_e:t_e+H_size] = \
                dense_2_sp_lists(mat,self.meas_idx(i),self.state_idx(i))
            t_e += H_size
        
        return sp.csr_matrix((data_l,(row_l,col_l)))

    def create_y(self, state_vec=None):
        '''
        Compute the residual vector.  Can compute it for the
        current state (pass in "None") or for a new state that you 
        want to test without setting the internal state.
        '''
        if state_vec is not None:
            state_data = self.vec_to_data(state_vec)
        else:
            state_data = self.states
        y = np.zeros(4*(self.N-1)+self.N)
        for i in range(1,self.N):
            # predicted measurement for dynamics is f(x_{k-1})-x_k
            pred_meas = self.prop_one_timestep(state_data[i-1])-state_data[i]
            y[self.dyn_idx(i):self.dyn_idx(i+1)]=self.S_Q_inv.dot(-pred_meas)
        # Now do the measurements received
        for i in range(self.N):
            pred_meas = atan2(state_data[i,1],state_data[i,0])
            tmp= self.meas[i]-pred_meas
            if tmp > pi:
                tmp -= 2*pi
            if tmp < -pi:
                tmp += 2*pi
            y[self.meas_idx(i):self.meas_idx(i+1)] = self.S_R_inv * tmp
        return y

    def vec_to_data(self,vec):
        going_out = np.zeros((self.N,4))
        for i in range(self.N):
            going_out[i] = vec[i*4:(i+1)*4]
        return going_out

    def add_delta(self,delta_x: np.array = None) -> np.array:
        '''
        This takes the current state and adds on delta_x.
        It DOES NOT  modify any internal state

        Inputs: delta_x, a self.N X 3 vector (np.array)h
        Returns: a full state vector of the same size
        '''
        going_out = np.zeros(self.N*4)
        if delta_x is None:
            delta_x = np.zeros(self.N*4)
        for i in range(self.N):
            going_out[i*4:(i+1)*4] = self.states[i]+ \
                delta_x[i*4:(i+1)*4]

        return going_out

    def update_state(self,delta_x):
        '''
        Changes the internal states data structure with the
        delta_x that is included here
        
        Returns nothing (only modifies internal state)
        '''
        for i in range(self.N):
            self.states[i] += delta_x[i*4:(i+1)*4]       

    def opt(self):
        '''
        Create the Jacobian matrix (L) and the residual vector (y) for
        the current state.  Find the best linear approximation to minimize y
        and move in that direction.  Repeat until convergence. 
        (This is a damped Gauss-Newton optimization procedure)
        '''
        finished=False
        num_iters=0
        while not finished:
            L= self.create_L()
            # print ('L shape is',L.shape)
            # plt.spy(L,markersize=1)
            # plt.show()
            # for i in range(9):
            #     print('For column',i)
            #     test_Jacobian(self,i,.0001)
            # test_Jacobian(self,2)
            
            y= self.create_y()
            M = L.T.dot(L)
            Lty = L.T.dot(y)
            delta_x = spla.spsolve(M,Lty)
            scale = 1
            # Damp the Gauss-Newton step if it doesn't do what the linearization predicts
            scale_good = la.norm(delta_x) < 10 # if the first step is too small, just do it and don't even check
            while not scale_good:
                next_y = self.create_y(self.add_delta(delta_x*scale))
                pred_y = y-L.dot(delta_x*scale)
                y_mag = y.T.dot(y)
                ratio = (y_mag - next_y.dot(next_y))/(y_mag-pred_y.dot(pred_y))
                if ratio < 4. and ratio > .25:
                    scale_good = True
                else:
                    scale /= 2.0
                    if scale < .001:
                        print('Your derivatives are probably wrong!  scale is',scale)
                assert(scale > 1E-6)
            num_iters+=1
            self.update_state(delta_x*scale)
            print('iteration',num_iters,'delta_x length was',la.norm(delta_x*scale), 'scale was',scale)
            finished = la.norm(delta_x)<12 or num_iters > 100


def test_Jacobian(batch_uni, col, dx = .001):
    '''
    This function is useful for debugging. If things don't work, try to 
    figure out where the derivative matrix is wrong.
    '''
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

    
if __name__ == '__main__':
    prefix='slide_example'
    data = np.load(f'{prefix}.npz')
    meas = data['meas']
    truth = data['truth']
    dt = data['dt']
    R = data['R']
    Q = data['Q']
    plt.plot(truth[:,0],truth[:,1],'r',label='truth')

    opt_class = satelliteModelBatch(meas, R, Q, dt)
    # print (" a sample F")
    # print(opt_class.F_mat(opt_class.states[10]))
    plt.plot(opt_class.states[:,0],opt_class.states[:,1],'g',label='orig')

    opt_class.opt()
    plt.plot(opt_class.states[:,0],opt_class.states[:,1],'b',label='opt')
    plt.legend()
    plt.savefig(f'{prefix}_res_new.png')
    
    plt.figure()
    plt.plot(opt_class.states[:,0],opt_class.states[:,1],c='b', label='estimate')
    plt.plot(truth[:,0],truth[:,1],'r--',label='truth')
    plt.legend()
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.savefig('FG_'+prefix+'_new.png')
    plt.show()


    plt.figure()
    plt.plot(opt_class.states-truth)
    plt.legend (['x','y','vx','vy'])
    plt.title('errors')
    plt.savefig(f'{prefix}_errors_new.png')
    plt.show()

    np.savez('fg_'+prefix+'_res_new',fg_res=opt_class.states, truth=truth)



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

