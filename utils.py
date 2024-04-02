import numpy as np
import matplotlib.pyplot as plt 
from abc import ABC,abstractmethod

class Bandit(ABC):

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


class bandit_arm_greedy(Bandit):

    def __init__(self, p, n_t,esp):
        self.p = p
        self.n_t=n_t
        self.esp=esp
        self.p_estimate = 0.
        self.N = 0 
   
    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x):
        
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N
         
    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'
    
    def experiment(self):
        bandits = [bandit_arm_greedy(p,self.n_t, self.esp) for p in self.p]

        rewards = np.zeros(self.n_t)
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        optimal_j = np.argmax([b.p for b in bandits])
        print(f'optimal bandit: {optimal_j}')
        
        for i in range(self.n_t):
            if np.random.random() < self.esp:
                num_times_explored += 1
                j = np.random.randint(len(bandits))
            else:
                num_times_exploited += 1
                j = np.argmax([b.p_estimate for b in bandits])

            if j == optimal_j:
                num_optimal += 1

            # pull the arm for the bandit with the largest sample
            x = bandits[j].pull()

            # update rewards log
            rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
            bandits[j].update(x)
        
        return bandits,rewards,num_times_explored, num_times_exploited, num_optimal
    
    def report(self):
        bandits,rewards,num_times_explored, num_times_exploited, num_optimal=self.experiment()
        for b in bandits:
            print(f"Mean Estimate: {b.p_estimate :.4f}")
    
        print(f"Total Reward Earned: {rewards.sum()}")
        print(f"Overall Win Rate: {rewards.sum() / self.n_t :.4f}")
        print(f"# of explored: {num_times_explored}")
        print(f"# of exploited: {num_times_exploited}")
        print(f"# of times selected the optimal bandit: {num_optimal}")


class bandit_arm_greedy_eps(Bandit):

    def __init__(self, m, n_t,esp):
        self.m = m
        self.n_t=n_t
        self.esp=esp
        self.m_estimate = 0
        self.N = 0 
   
    def pull(self):
        return np.random.randn() < self.m
    
    def update(self, x):
        
        self.N += 1
        self.m_estimate = (1 - 1.0/self.N)*self.m_estimate + 1.0/self.N*x
         
    def __repr__(self):
        return f'An Arm with {self.m} Win Rate'
    
    def experiment(self):
        bandits = [bandit_arm_greedy_eps(p,self.n_t, self.esp) for p in self.m]

        means = np.array(self.m) # count number of suboptimal choices
        true_best = np.argmax(means)  
        count_suboptimal = 0

        data = np.empty(self.n_t)

        for i in range(self.n_t):
        
            p = np.random.random()
            if p < self.esp:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.m_estimate for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)

            if j != true_best:
                count_suboptimal += 1

            # for the plot
            data[i] = x

        cumulative_average = np.cumsum(data) / (np.arange(self.n_t) + 1)
        estimated_avg_rewards=[round(b.m_estimate,3) for b in bandits]
        print(f'Estimated average reward where epsilon= {self.esp}:---{estimated_avg_rewards}')
        print(f'Percent suboptimal where epsilon={self.esp}:---{float(count_suboptimal)/ self.n_t}')
        print("--------------------------------------------------")
        return cumulative_average
    
    def report(self):
        pass
