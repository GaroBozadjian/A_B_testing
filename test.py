from utils import *
import matplotlib.pyplot as plt
import numpy as np


NUM_TRIALS = 100000
BANDIT_RETURN = [1.5, 2.5, 3.5]
EPS=[0.1,0.05,0.01]

# test= bandit_arm(BANDIT_PROBABILITIES,NUM_TRIALS,EPS)

# test.experiment()

# test=bandit_arm_eps(BANDIT_RETURN,NUM_TRIALS,EPS)

[bandit_arm_greedy_eps(BANDIT_RETURN,NUM_TRIALS,i).experiment() for i in EPS]