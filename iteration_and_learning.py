
#!/Users/sidharthsharma/opt/anaconda3/bin/python

# All imports
import numpy as np
import matplotlib.pyplot as plt
import random

## Requirements
    # Implement Value Iteration and Policy Iteration
    # to obtain the optimal value of Q* and policy π*
    # Consider Infinite horizon MDP
    # |S| = 64
    # |A| = 4
    # Discount factor = 0.9
class MDP:
    def __init__(self,debug):
        self.state_space = 64
        self.action_space = 4
        self.discount_factor = 0.9
        self.transition_m = np.load('P.npy')
        self.reward_m    = np.load('R.npy')
        self.debug = debug
        if self.debug:
            print("This is the init")

    def data_check(self):
        if self.debug:
            print("In datacheck")
        if self.transition_m.any():
            print("Transition Probability data loaded. Shape: {}".format(self.transition_m.shape))
        if self.reward_m.any():
            print("Reward data loaded. Shape: {}".format(self.reward_m.shape))

    def bellman_operator(self,Q):
        # What should be the dimension of Q(s,a)?
        print("Q shape bellman: ",np.amax(np.dot(self.transition_m,Q),axis=2).shape)
        QT = self.reward_m + self.discount_factor * np.dot(self.transition_m,np.amax(Q,axis=1)) # Dimensions: (64x4) + 1 * (64x4x64)
        return QT
    
    def value_iteration(self,T):
        '''
        Return: Error from value iteration 
        # Q(t+1) = Tb(Q(t))
        '''
        # Initalization 
        # Q can be initalized from [0,1/(1-gamma))
        vi_error: list = [] 
        Q_function = np.zeros((self.state_space,self.action_space))             # Dimension: 64 x 4
        #Q_function_prev = np.zeros((self.state_space,self.action_space))       # Dimension: 64 x 4
        
        Q_function_values:list = []
        # Iteration 
        # Iterate till convergence Q(t+1) = TQ(t)(s,a)
        for i in range(T):
            Q_function_values.append(Q_function)
            Q_function = self.bellman_operator(Q_function)
        
        for t in range(T-1):
            error = np.log(np.linalg.norm(Q_function_values[T-1] - Q_function_values[t],np.Inf))         # Dimension: 64 x 4
            vi_error.append(error)
        return vi_error
    
    def policy_iteration(self,T):
        '''
        Return: Error from policy iteration 
        '''
        pi_error: list = []
        Q_function_values:list = []
        policies:list = []
        Q_function = np.zeros((self.state_space,self.action_space))                     # Dimension: 64 x 4
        policy = np.zeros((self.state_space), dtype=int)                                       # Dimension: 64 -> Best action for a each state 

        # Iteration 
        # Every Iteration we get a new policy π = [π0,π1,π2,π3....]
        for m in range(T):
            policies.append(policy)
            Q_function_values.append(Q_function)
            # Policy Evaluation: Via linear programming
            indices = np.arange(64)
            #print(m)
            # print(policy.dtype)
            # self.transition_m = self.transition_m .transpose(0, 2, 1)
            P = self.transition_m[indices,policy.flatten(),:]
            #print(P.shape)
            Q_function = self.reward_m + self.discount_factor * np.dot(P, Q_function)   # 64 x 4 x 64 
            #print('Q function: ',Q_function.shape)                                                                                                        # 64 x 1 x 64 (Basically the action of each state is given by the policy)
            # Policy Improvement: In a  greedy way we select the action that maximises Q value
            policy = np.argmax(Q_function,axis=1).astype(int)
            #print('Policy Shape: ',policy.shape)  

        for t in range(T-1):
            error = np.log(np.amax(np.abs(Q_function_values[T-1] - Q_function_values[t]),axis=0))        # Dimension: 64 x 4
            pi_error.append(error)
        return pi_error,Q_function_values[T-1]
    
    
    def q_learning(self,T,lr,Q_star,q=1):
        '''
        Return: Error from value iteration 
        # Q(t+1) = Tb(Q(t))
        '''
        # Initalization 
        ql_error: list = [] 
        Q_function = np.zeros((self.state_space,self.action_space))             # Dimension: 64 x 4
        Q_function_values:list = []
        # Iteration 
        # Iterate till convergence Q(t+1) = TQ(t)(s,a)
        for i in range(T):
            if q == 2:
                lr = 1/(1+(1-self.discount_factor)*(i+1))
            Q_function_values.append(Q_function)
            # get the value of the next state form  the current state
            # What is q function storing - the value that can be used to do something 
            # Something what - what is it that it can do???
            # it is a 64x4 matrix 
            s_new  = random.sample(range(64), k=64)

            Q_function = (1-lr)*Q_function + lr*(self.reward_m + self.discount_factor*(np.max(Q_function[s_new])))
        
        for t in range(T-1):
            error = np.log(np.linalg.norm(Q_star - Q_function_values[t],np.Inf))         # Dimension: 64 x 4
            ql_error.append(error)
        return ql_error

    def error_plot(self,x,y1,y2):
        # create the figure and axis
        fig, ax = plt.subplots()

        ax.plot(x,y1,label='Value Iteration')
        ax.plot(x,y2,label='Policy Iteration')
        
        ax.legend()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('(log (∥Q(T)VI − Q(t)VI ∥∞))')
        ax.set_title('Two lines on the same plot')

        #plt.xlabel('Iteration')
        #plt.ylabel('Error')
        #plt.title('Error Plot')
        # display the plot
        plt.show()


    def q_learning_error_plot(self,x,y1,y2,y3,y4):
        # create the figure and axis
        fig, ax = plt.subplots()

        ax.plot(x,y1,label='lr=0.01')
        ax.plot(x,y2,label='lr=0.05')
        ax.plot(x,y3,label='lr=0.1')
        ax.plot(x,y4,label='lr=partb')
        
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Estimation Error')
        ax.set_title('Error Plot')

        #plt.xlabel('Iteration')
        #plt.ylabel('Error')
        #plt.title('Error Plot')
        # display the plot
        plt.show()

def main():
    # Initalize and data check 
    mdp = MDP(0)
    mdp.data_check()
    T = 100
    
    plot_list = [i for i in range(99)]
    ## Take the first 100 values from this yield of VI
    vi_100 = mdp.value_iteration(T)
    
    ## Take the first 100 values from this yield of PI
    pi_100,Q_star = mdp.policy_iteration(T)
    ## Plot the log error of PI
    #print(pi_100)
    mdp.error_plot(plot_list,vi_100,pi_100)
    
    np.random.seed(3)
    '''
    q_1 = mdp.q_learning(2000,0.01, Q_star)
    q_2 = mdp.q_learning(2000,0.05, Q_star)
    q_3 = mdp.q_learning(2000,0.1,  Q_star)
    q_4 = mdp.q_learning(2000,0.1,  Q_star,q=2)
    
    plot_list2 = [i for i in range(2000)]
    mdp.q_learning_error_plot(plot_list2,q_1,q_2,q_3,q_4)
    '''
    
        
if __name__ == "__main__":
    main()