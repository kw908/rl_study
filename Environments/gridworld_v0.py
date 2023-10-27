## grid world from Sutton and Barto example 4.1

## use tuples to denote states (grid positions)
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from copy import copy

class environment():
  def __init__(self,sidelength,episodes) -> None:
      self.sidelength = sidelength
      self.l = sidelength-1
      self.actions = ["up","down","left","right"]
      self.states = []
      for i in range(self.sidelength):
        for j in range(self.sidelength):
            self.states.append(state([i,j],episodes+1))
      
      self.terminal_states = [[0,0],[self.l,self.l]]
      
      """For rendering"""
      self.width, self.height = 200, 200  # Set the width and height of the image
      self.cell_size = self.width // self.sidelength  # Calculate cell size for 4x4 grid
      self.image = Image.new('RGB', (self.width, self.height), 'white')
      self.draw = ImageDraw.Draw(self.image)
      self.font = ImageFont.load_default()  # You can also specify a specific font if needed
      self.arrows = {"up":"\u2191",
                    "down":"\u2193",
                    "right":"\u2192",
                    "left":"\u2190"}
    

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
       return next_state_value, reward 

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
      return next_state_value, reward
    
    else:
        for pos in next_state_alias:
            if pos < 0 or pos > self.l: #off the grid case
                reward = -1
                next_state_alias = copy(state.alias)
                next_state_value = copy(state.value[episode])
                return next_state_value, reward
        
        reward = -1
        i = self.pair_to_number(next_state_alias)
        next_state_alias = self.states[i].alias
        next_state_value = self.states[i].value[episode]
        return next_state_value, reward
    

  def render(self):
      for i in range(self.sidelength): # column
        for j in range(self.sidelength): # row
            cell_x, cell_y = i * self.cell_size, j * self.cell_size
            
            cell_number = self.pair_to_number([j,i])
            
            self.draw.rectangle([cell_x, cell_y, cell_x + self.cell_size, cell_y + self.cell_size], outline='black')  # Draw cell border
            text_width, text_height = self.draw.textsize(str(cell_number), self.font)  # Calculate text size
            text_x = cell_x + (self.cell_size - text_width) // 2
            text_y = cell_y + (self.cell_size - text_height) // 2
            # text_y -= 3
            
            if [j,i] in self.terminal_states:
               self.draw.multiline_text((text_x-3, text_y), "term", font=self.font, fill=(0, 0, 0))
            else:
                for a in ["up","down","left","right"]:
                    if self.states[cell_number].policy[a][0] != 0:
                        self.draw.multiline_text((text_x-3, text_y), a, font=self.font, fill=(0, 0, 0))
                        text_y+=text_height
                    
            

                # self.draw.text((text_x, text_y), a, fill='black', font=self.font)
      self.image.show()

class agent():
   def __init__(self) -> None:
      self.state = state([0,0])

class state():
   def __init__(self,position:[int,int],episode:int) -> None:
      self.alias = position
      self.value = np.zeros(episode+1)
      self.policy = {"up":[0.25,1.0,0.0], "down":[0.25,1.0,0.0], "left":[0.25,1.0,0.0], "right":[0.25,1.0,0.0]} 
      #{action:[p(a), p(s',r|s,a)=1, q_value]

