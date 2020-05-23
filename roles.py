import numpy as np
import pandas as pd
from random import randint

def updateQ(prev_q, next_q, reward, alpha=0.2, discount_factor=0.2):
    return (1 - alpha) * prev_q + alpha * (reward + discount_factor * next_max_q)

class Player:
    def __init__(self, num_total, num_states=16, price_range=(5, 20)):
        self.min_price, self.max_price = price_range
        self.num_sold = 0
        self.num_clicked = 0
        self.current_price = 0
        # shape: num_clicked, num_sold, num_states, num_price
        self.q_table = np.zeros([num_total, num_total, num_states, self.max_price - self.min_price + 1])

    def setQ(self, is_clicked, is_sold, state, reward, alpha=0.2, discount_factor=0.2):
        prev_q = q_table[self.num_clicked, self.num_sold, state, self.current_price]
        next_max_q = np.max(q_table[self.num_clicked + is_clicked, self.num_sold + is_sold])
        q_table[self.num_clicked, self.num_sold, state, self.current_price] = updateQ(pewv_q, next_max_q, reward, alpha=0.2, discount_factor=0.2)

    def updateStates(self, is_clicked, is_sold, rank, state):
        reward = 10 # some formula involing current_price and rank
        self.setQ(is_clicked, is_sold, state, self.current_price, reward)
        slef.num_clicked += is_clicked
        self.num_sold += is_sold
    
    def getPrice(self, state):
        self.current_price = np.argmax(q_table[self.num_clicked, self.num_sold, state]) + self.min_price
        return self.current_price

class Auctioneer:
    def __init__(self, n_total_rounds, p_vehicle, p_driver, p_insured, p_marital, n_players = 5, recording = False):
        self.p_vehicle = p_vehicle
        self.p_driver = p_driver
        self.p_insured = p_insured
        self.p_marital = p_marital
        self.n_players = n_players
        self.auction_result = []
        self.recording = recording
        self.players = [Player(n_total_rounds) for _ in range(n_players)]

    def getRadnomImpression(self):
        # get a random person with (v, d, m, i)
        return 8 * v + 4 * d + 2 * m + i

    def generateResult(self, state, ranking):
        # need some algorithm to generate the result, right now, whoever bid highest get the policy sold
        # -1 for no policy sold
        return ranking[0], ranking[:3], ranking[3:]
    
    def auction(self):
        state = self.getRadnomImpression()
        price = [player.getPrice(state) for player in self.players]
        ranking = self.getSortedPlayerIndex(price)
        sold, clicked, not_clicked = self.generateResult(state, ranking)
        self.players[sold].updateStates(1, 1, ranking.index(sold), state)
        for i in clicked:
            self.players[i].updateStates(1, 0, ranking.index(i), state)
        for i in not_clicked:
            self.players[i].updateStates(0, 0, ranking.index(i), state)
        if self.recording:
            self.auction_result.append({'price':price, 'ranking': ranking, 'sold':sold, 'clicked':clicked, 'not_clicked':not_clicked})

    def getSortedPlayerIndex(self, price):
        # need some way to deal with the same price
        return sorted(range(self.n_players), key = price.__getitem__)
