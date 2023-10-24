## grid world from Sutton and Barto example 4.1

## use tuples to denote states (grid positions)
import math
import numpy as np
from copy import deepcopy

class environment():
  def __init__(self,sidelength) -> None:
    self.sidelength = sidelength
    self.l = sidelength-1
    self.action = ["up","down","left","right"]
    self.states = []
    for i in range(self.sidelength):
       for j in range(self.sidelength):
          self.states.append(state([i,j]))
    
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
    

  def step(self,state,action):

    next_state = deepcopy(state)
    reward = 0

    if action == "up":
        next_state.alias[0] -= 1
    if action == "down":
        next_state.alias[0] += 1
    if action == "left":
        next_state.alias[1] -= 1
    if action == "right":
        next_state.alias[1] += 1
    
    if next_state.alias in self.terminal_states:
      reward = -1 
      next_state.value = 0
      return next_state, reward
    
    else:
        for pos in next_state.alias:
            if pos < 0 or pos > self.l: #off the grid case
                reward = -1
                next_state.alias = state.alias
                next_state.value = state.value
                return next_state, reward
            else: 
                reward = -1
                i = self.pair_to_number(next_state.alias)
                next_state.value = self.state[i]
                return next_state, reward
    

  def render(self):
       pass

class agent():
   def __init__(self) -> None:
      self.state = state([0,0])

class state():
   def __init__(self,position:[int,int]) -> None:
      self.alias = position
      self.value = 0
