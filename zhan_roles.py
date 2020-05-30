import numpy as np
import pandas as pd
from random import randint, shuffle, random
from collections import Counter

def updateQ(prev_q, next_q, reward, alpha=0.2, discount_factor=0.2):
    return (1 - alpha) * prev_q + alpha * (reward + discount_factor * next_q)

class Company:
    '''The class of all companies'''
    def __init__(self, num_total, updateQ, num_states=16, price_range=(5, 20)):
        """Initialization function

        Parameters:
            num_total (int):     Number of total rounds
            updateQ (function):  Function for updating the q_table
            num_states (int):    default 16 states (8 * v + 4 * d + 2 * m + i)
            price_range (tuple): the tuple of (min_price, max_price), default (5, 20)

        Returns:
            None
        """
        self.min_price, self.max_price = price_range
        self.num_sold = 0      # number of policies sold currently
        self.num_clicked = 0   # number of impressions clicked currently
        self.test = False
        self.num_total_sold = 0
        self.num_total = num_total
        self.current_price = 0 # bidding price for current customer
        self.updateQ = updateQ
        self.record = {'click':[], 
                                    'currently_insured':[], 
                                    'number_of_vehicles':[],
                                    'number_of_drivers':[], 
                                    'rank':[], 
                                    'policies sold':[], 
                                    'married':[]}

        # shape: num_clicked  x  num_sold  x  num_states  x  num_price(actions)
        self.q_table = np.zeros([num_total + 1, num_total + 1, num_states, self.max_price - self.min_price + 1])
        
    def reset(self):
        self.num_sold = 0
        self.num_clicked = 0
        self.current_price = 0

    def setQ(self, is_clicked, is_sold, state, reward, alpha=0.2, discount_factor=0.2):
        """set Q value

        Parameters:
            is_clicked (int):        0 for not clicked, 1 for clicked
            is_sold (int):           0 for not clicked, 1 for clicked
            state (int):             0-15, current customer's state
            reward (int):            the reward resulting from the bidding price
            alpha (float):           0-1, hyperparameter for calculating Q-value
            discount_factor (float): 0-1 hyperparameter for calculating Q-value

        Returns:
            None
        """
        prev_q = self.q_table[self.num_clicked, self.num_sold, state, self.current_price]
        next_max_q = np.max(self.q_table[self.num_clicked + is_clicked, self.num_sold + is_sold])
        self.q_table[self.num_clicked, self.num_sold, state, self.current_price] = self.updateQ(prev_q, next_max_q, reward, alpha=0.2, discount_factor=0.2)

    def updateStates(self, is_clicked, is_sold, rank, state):
        """update q_table, num_sold, num_clicked; calculating reward

        Parameters:
            is_clicked (int): 0 for not clicked, 1 for clicked
            is_sold (int):    0 for not clicked, 1 for clicked
            rank (int):       the rank of this bid.
            state (int):      0-15, current customer's state

        Returns:
            None
        """
        if self.test:
            self.num_clicked += is_clicked
            self.num_clicked %= self.num_total
            self.num_sold += is_sold
            self.num_sold %= self.num_total
            self.num_total_sold += is_sold
            self.record['click'].append(is_clicked)
            self.record['currently_insured'].append(state % 2)
            self.record['number_of_vehicles'].append(state // 8)
            self.record['number_of_drivers'].append((state // 4) % 2)
            self.record['rank'].append(rank)
            self.record['policies sold'].append(is_sold)
            self.record['married'].append((state // 2) % 2)
            return
        reward = - self.current_price * is_clicked + 10 * self.max_price * is_sold
        self.setQ(is_clicked, is_sold, state, self.current_price, reward)
        self.num_clicked += is_clicked
        self.num_sold += is_sold

    def getPrice(self, state, epsilon = 0.1):
        """generate price for this bid

        Parameters:
            state (int): 0-15, current customer's state

        Returns:
            int: the price for this customer
        """
        if random() < epsilon:
            self.current_price = randint(0, self.max_price - self.min_price)
        else:
            self.current_price = np.argmax(self.q_table[self.num_clicked, self.num_sold, state])
        return self.current_price + self.min_price

class SearchWebsite:
    '''The class of the searching website'''
    def __init__(self, updateQ, n_total_rounds, p_vehicle, p_driver, p_insured, p_marital, n_companies = 5, recording = False):
        """Initialization function

        Parameters:
            n_total_rounds (int): number of total rounds
            p_vehicle (float): probability that a customer has two cars
            p_driver (float): probability that a customer has two drivers
            p_insured (float): probability that a customer is insured
            p_marital (float): probability that a customer is married
            n_companies (int): number of companies
            recording (boolean): whether to record every impression result.

        Returns:
            None
        """
        self.p_vehicle = p_vehicle
        self.p_driver = p_driver
        self.p_insured = p_insured
        self.p_marital = p_marital
        self.n_companies = n_companies
        self.auction_result = []
        self.recording = recording

        # the list of all companies, each company is represented by its index,
        # i.e. self.companies[0] represents Company 0
        self.companies = [Company(n_total_rounds,  updateQ) for _ in range(n_companies)]
        
        self.customer_click_prob = [
            [0.571429, 0.200000, 0.100000, 0, 0],       #0
            [0.636364, 0.250000, 0, 0.071429, 0],       #1
            [0.509554, 0.215385, 0.147679, 0.038217, 0],#2
            [0.500000, 0.137255, 0.147321, 0.059211, 0],#3
            [0.514184, 0.206897, 0.194690, 0, 0],       #4
            [0.523364, 0.189944, 0.136612, 0.050847, 0],#5
            [0.488746, 0.161491, 0.087591, 0, 0],       #6
            [0.582524, 0.208589, 0.185567, 0.037313, 0],#7
            [0, 0, 0.047619, 0.121212, 0.017544],       #8
            [0, 0, 0.294118, 0.080000, 0],              #9
            [0, 0, 0.167568, 0.031546, 0.024777],       #10
            [0, 0, 0.162963, 0.046053, 0.016299],       #11
            [0, 0.175258, 0.125000, 0.031008, 0.010204],#12
            [0, 0.137931, 0.143939, 0.036765, 0.020833],#13
            [0, 0.193548, 0.123077, 0.039370, 0.022989],#14
            [0, 0.231884, 0.167939, 0.033784, 0.033333]]
        self.customer_buy_prob = [[0.8974358974358975, 0, 0.05128205128205128, 0.05128205128205128, 0, 0],
 [0.8205128205128205,
  0,
  0.10256410256410256,
  0.05128205128205128,
  0,
  0.02564102564102564],
 [0.8820375335120644,
  0,
  0.06032171581769437,
  0.030831099195710455,
  0.020107238605898123,
  0.006702412868632707],
 [0.9128065395095368,
  0,
  0.04632152588555858,
  0.012261580381471389,
  0.02316076294277929,
  0.005449591280653951],
 [0.8154657293497364,
  0,
  0.1335676625659051,
  0.033391915641476276,
  0.01757469244288225,
  0],
 [0.9471890971039182,
  0,
  0.027257240204429302,
  0.015332197614991482,
  0.008517887563884156,
  0.0017035775127768314],
 [0.8620689655172413,
  0,
  0.11494252873563218,
  0.019704433497536946,
  0.003284072249589491,
  0],
 [0.9511784511784511,
  0,
  0.021885521885521887,
  0.0101010101010101,
  0.015151515151515152,
  0.0016835016835016834],
 [0.9819819819819819, 0, 0, 0, 0.009009009009009009, 0.009009009009009009],
 [0.978494623655914, 0, 0, 0, 0.021505376344086023, 0],
 [0.9811227024341779, 0, 0, 0, 0.014903129657228016, 0.003974167908594138],
 [0.9888132295719845, 0, 0, 0, 0.00632295719844358, 0.0048638132295719845],
 [0.96900826446281,
  0,
  0,
  0.014462809917355372,
  0.012396694214876033,
  0.004132231404958678],
 [0.9800443458980045,
  0,
  0,
  0.006651884700665188,
  0.008869179600886918,
  0.004434589800443459],
 [0.9725400457665904, 0, 0, 0.016018306636155607, 0.011441647597254004, 0],
 [0.9817351598173516,
  0,
  0,
  0.0045662100456621,
  0.0091324200913242,
  0.0045662100456621]]

    def generateRandomCustomer(self,p_vehicle, p_driver, p_insured, p_marital):
        """Generate a random customer

        Parameters:
            p_vehicle (float): probability that a customer has two cars
            p_driver (float): probability that a customer has two drivers
            p_insured (float): probability that a customer is insured
            p_marital (float): probability that a customer is married

        Returns:
            int: an int between 0 and 15 representing a customer's state.

        Note:
            The formula is  8 * v + 4 * d + 2 * m + i
        """
        v = np.random.binomial(1, p_vehicle)
        d = np.random.binomial(1, p_driver)
        m = np.random.binomial(1, p_marital)
        i = np.random.binomial(1, p_insured)
        return 8 * v + 4 * d + 2 * m + i

    def getResult(self, state, ranking):
        """Generating the result based on state and ranking

        Parameters:
            state (int): 0-15, current customer's state
            ranking (list(int)): the list of indexes of companies, representing their price ranking

        Returns:
            (int, list(int), list(int)): (sold, clicked, not_clicked) where
                                          sold: the index of company who sold its policy, none if no company get its policy sold
                                          clicked: the list of indexes of companies who got its impression clicked
                                          not_clicked: the rest companies
        """
        # need some algorithm to generate the result, right now, whoever bid highest get the policy sold
        # None for no policy sold
        clicked = []
        not_clicked = []
        for i in range(self.n_companies):
            t = np.random.binomial(1, self.customer_click_prob[state][i])
            if t == 1:
                clicked.append(ranking[i])
            else:
                not_clicked.append(ranking[i])
        
        sold = np.random.choice(np.arange(-1, 5), p=self.customer_buy_prob[state])
        return sold, clicked, not_clicked

    def train(self):
        for _ in range(100):
            self.auction()
        self.reset()
    
    def reset(self):
        self.auction_result = []
        for company in self.companies:
            company.reset()
        
        
    def auction(self):
        """auction process

        Parameters:
            None

        Returns:
            None
        """

        # get a random customer
        state = self.generateRandomCustomer(self.p_vehicle, self.p_driver, self.p_insured, self.p_marital)

        # pass the customer info to each company, let them return the bidding price
        price = [company.getPrice(state) for company in self.companies]

        # get the ranking of these companies based on prices
        ranking = self.getSortedCompanyIndex(price)

        # grab the result based the customer and ranking
        sold, clicked, not_clicked = self.getResult(state, ranking)

        # let each company know the result
        if sold != -1:
            self.companies[sold].updateStates(1, 1, ranking.index(sold), state)
        for i in clicked:
            if i != sold:
                self.companies[i].updateStates(1, 0, ranking.index(i), state)
        for i in not_clicked:
            self.companies[i].updateStates(0, 0, ranking.index(i), state)

        # if recording is on, record it
        if self.recording:
            self.auction_result.append({'price':price, 'ranking': ranking, 'sold':sold, 'clicked':clicked, 'not_clicked':not_clicked})

    def getSortedCompanyIndex(self, price):
        """Sort the companies

        Parameters:
            price (list(int)): the list of prices for each company

        Returns:
            list(int): sorted indexes
        """
        indexes = list(range(self.n_companies))
        shuffle(indexes)

        return sorted(indexes, key = price.__getitem__, reverse = True)
    
    def summary(self):
        if self.recording:
            total_sold = [0] * self.n_companies
            for item in self.auction_result:
                total_sold[item['sold']] += 1
            print(total_sold)
