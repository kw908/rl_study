## Black jack for Monte Carlo approach ##

import math
import numpy as np
import random
import itertools as iter

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

        self.rewards = []

        r = [-1, 0, 1]
        dealer_showing = [1,2,3,4,5,6,7,8,9,10]
        player_sum = [12,13,14,15,16,17,18,19,20,21]
        self.states = list(iter.product(r, dealer_showing, player_sum)) #200 states total

    def reset(self):
        self.dealer_cards = []
        self.player_cards = []
        self.rewards = []

    def hits(self): #draw a card
        random.seed()
        i = random.randint((1,52))
        self.step+=1
        return self.deck[i]

    
    def dealer(self):
        s = sum(self.dealer_cards)
        if s>=17 and s<=21: 
            return s, "sticks"
        elif s<17:
            self.dealer_cards.append(self.hits())
            return "hits"
        else: return "busted"
    
    def agent(self): #player with policy
        s = sum(self.player_cards)
        while (s<20):
            self.player_cards.append(self.hits())
            self.rewards.append(0) #in-game reward is 0
            if self.player_cards.count(1) == 1:
                self.check_usable()
            s = sum(self.player_cards)
            if s>21: return "busted"
        return "sticks"
    
    def check_usable(self):
        i = self.player_cards.index(1,0,2) 
        self.player_cards[i] = 11
        if sum(self.player_cards) >21:
            self.player_cards[i] = 1
            

    def game(self): #one game is one episode
        # deal two cards to each
        self.dealer_cards.append(self.hits())
        self.dealer_cards.append(self.hits())
        self.player_cards.append(self.hits())
        self.player_cards.append(self.hits())

        if self.player_cards.count(1) == 1:
            self.check_usable()
            
        if sum(self.player_cards) == 21: #natural
            if sum(self.dealer_cards) == 21:
                term_reward = 0 
                return term_reward
            else:
                term_reward = 1 #player won
                return term_reward

        #keep going if no results
        result = self.agent()
        if result == "sticks":
            pass
        elif result == "busted": #player lost
            term_reward = -1
            return term_reward
        
        result = self.dealer()
        if result == "sticks":
            pass
        elif result == "busted": #player won
            term_reward = 1
            return term_reward
        
        if sum(self.player_cards)>sum(self.dealer_cards):
            term_reward = 1
        elif sum(self.player_cards)<sum(self.dealer_cards):
            term_reward = -1
        else: term_reward = 0 #tie

        return term_reward

    def render(self):
        pass
    
            

