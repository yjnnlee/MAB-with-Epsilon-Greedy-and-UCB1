import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#Make slot arm
class SlotArm():
  def __init__(self, p):
    self.p = p #Probability of getting a coin
  
  #Get a reward for choosing an arm
  def draw(self):
    if self.p > random.random():
      return 1.0
    else:
      return 0.0
    
#Calculation epsilon-Greedy
class EpsilonGreedy():
  #init e-greedy
  def __init__(self, epsilon):
    self.epsilon = epsilon #Probability of exploration

  #init n, v (n_arms: number of arms)
  def initialize(self, n_arms): 
    self.n = np.zeros(n_arms) #number of attempts for each arm - list
    self.v = np.zeros(n_arms) #value of each arm - list (평균보상)

  #Select arm
  def select_arm(self):
    if self.epsilon > random.random():
      return np.random.randint(0, len(self.v)) #select random
    else:
      return np.argmax(self.v) #select high value arm

  #Update parameter
  def update(self, chosen_arm, reward, t):
    self.n[chosen_arm] += 1 #number of attempts of the selected arm + 1

    #update the value of the selected arm
    n = self.n[chosen_arm]
    v = self.v[chosen_arm]
    self.v[chosen_arm] = ((n-1) / float(n)) * v + (1 / float(n)) * reward

  #get string info
  def label(self):
    return 'E-greedy("+str(self.epsilon)+")'
  
  #run
def play(algo, arms, num_sims,num_time):
  times = np.zeros(num_sims*num_time) #game times
  rewards = np.zeros(num_sims*num_time) #rewards

  #roop simulation
  for time in range(num_time):
    algo.initialize(len(arms))

    #roop game
    for sim in range(num_time):
      #calculate index
      index = sim * num_time + time

      times[index] = time + 1
      chosen_arm = algo.select_arm()
      reward = arms[chosen_arm].draw()
      rewards[index] = reward

      #update parameter
      algo.update(chosen_arm, reward, time+1)

  return [times, rewards]

#select arm
arms = (SlotArm(0.3), SlotArm(0.5), SlotArm(0.9))

#set algorithm
algo = EpsilonGreedy(0.1)

#run
result = play(algo, arms, 1000, 250)

#draw graph
df = pd.DataFrame({'times': result[0], 'rewards': result[1]})
mean = df['rewards'].groupby(df['times']).mean()
plt.plot(mean, label = algo.label())

plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend(loc = 'best')
plt.show()
