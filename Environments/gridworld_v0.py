## grid world from Sutton and Barto example 4.1

## use tuples to denote states (grid positions)
import math
import numpy as np
from copy import copy

class environment():
  def __init__(self,sidelength,episode) -> None:
    self.sidelength = sidelength
    self.l = sidelength-1
    self.actions = ["up","down","left","right"]
    self.states = []
    for i in range(self.sidelength):
       for j in range(self.sidelength):
          self.states.append(state([i,j],episode))
    
    self.terminal_states = [[0,0],[self.l,self.l]]
  
  def number_to_pair(self,input:int): # convert from 1 to 14 notation to pair notation
     # input 1 to 14
     x = int(math.floor(input/self.sidelength))
     y = int(input % self.sidelength)
     return [x,y]
  
  def pair_to_number(self,input:[int,int]): # the other way around
     # input a number pair
     return input[0]*self.sidelength + input[1]

#   def reset(self,start_state):
#     self.agent_state = start_state
    

  def step(self,state,action,episode):

    next_state_alias = copy(state.alias)
    next_state_value = copy(state.value[episode])
    reward = 0

    if state.alias in self.terminal_states:
       reward = 0
       next_state_value = 0
       return next_state_alias, next_state_value, reward

    if action == "up":
        next_state_alias[0] -= 1
    if action == "down":
        next_state_alias[0] += 1
    if action == "left":
        next_state_alias[1] -= 1
    if action == "right":
        next_state_alias[1] += 1
    
    if next_state_alias in self.terminal_states:
      reward = -1 
      next_state_value = 0
      return next_state_alias, next_state_value, reward
    
    else:
        for pos in next_state_alias:
            if pos < 0 or pos > self.l: #off the grid case
                reward = -1
                next_state_alias = copy(state.alias)
                next_state_value = copy(state.value[episode])
                return next_state_alias, next_state_value, reward
            else: 
                reward = -1
                i = self.pair_to_number(next_state_alias)
                next_state_alias = self.states[i].alias
                next_state_value = self.states[i].value[episode]
                return next_state_alias, next_state_value, reward
    

  def render(self):
       pass

class agent():
   def __init__(self) -> None:
      self.state = state([0,0])

class state():
   def __init__(self,position:[int,int],episode:int) -> None:
      self.alias = position
      self.value = np.zeros(episode+1)
