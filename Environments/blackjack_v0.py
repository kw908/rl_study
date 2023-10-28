## Black jack for Monte Carlo approach ##

import math
import numpy as np
import random

"""for rendering"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


## Rewards: win=+1, lose=-1, draw=0, in-game=0
class environment():
    def __init__(self) -> None:
        self.deck = []
        for i in range(1,14):
            for _ in range(4):
                if i<10:
                    self.deck.append(i)
                else:
                    self.deck.append(10)
        
        self.dealer_cards = []
        self.player_cards = []

    def reset(self):
        self.dealer_cards = []
        self.player_cards = []

    def hits(self): #draw a card
        random.seed()
        i = random.randint((1,52))
        return self.deck[i]

    
    def dealer(self):
        s = sum(self.dealer_cards)
        if s>=17: 
            return s, "sticks"
        else:
            self.dealer_cards.append(self.hits())
            return "hits"

    def render(self):
        pass
    
            

