import numpy as np
import pandas as pd
from random import randint, shuffle, random
from collections import Counter

def updateQ(prev_q, next_q, reward, alpha=0.2, discount_factor=0.2):
    return (1 - alpha) * prev_q + alpha * (reward + discount_factor * next_q)

class Company:
    '''The class of all companies'''
    def __init__(self, updateQ, num_total=100, num_states=16, price_range=(5, 20)):
        """Initialization function

        Parameters:
            updateQ (function):  Function for updating the q_table
            num_total (int):     Number of total rounds
            num_states (int):    default 16 states (8 * v + 4 * d + 2 * m + i)
            price_range (tuple): the tuple of (min_price, max_price), default (5, 20)

        Returns:
            None
        """
        self.min_price, self.max_price = price_range
        self.num_sold = 0      # number of policies sold currently
        self.num_clicked = 0   # number of impressions clicked currently
        self.train = True
        self.num_total_sold = 0
        self.num_total_runs = 0
        self.num_total = num_total
        self.current_price = 0 # bidding price for current customer
        self.updateQ = updateQ
        self.record = {'click':[], 
                       'currently_insured':[], 
                       'number_of_vehicles':[],
                       'number_of_drivers':[], 
                       'rank':[], 
                       'policies sold':[], 
                       'married':[],
                       'current_price':[]}

        # shape: num_clicked  x  num_sold  x  num_states  x  num_price(actions)
        self.q_table = np.zeros([num_total + 1, num_total + 1, num_states, self.max_price - self.min_price + 1])
        
    def reset(self):
        self.num_total_runs = 0
        self.num_sold = 0   
        self.num_clicked = 0
        self.num_total_sold = 0
        self.record = {'click':[], 
                       'currently_insured':[], 
                       'number_of_vehicles':[],
                       'number_of_drivers':[], 
                       'rank':[], 
                       'policies sold':[], 
                       'married':[],
                       'current_price':[]}

    def updateQTable(self, is_clicked, is_sold, state, reward, alpha=0.2, discount_factor=0.2):
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
        next_max_q = np.max(self.q_table[(self.num_clicked + is_clicked) % self.num_total, 
                                         (self.num_sold + is_sold) % self.num_total])
        
        # update Q table
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
        if self.train:
            reward = - self.current_price * is_clicked + 10 * self.max_price * is_sold
            self.updateQTable(is_clicked, is_sold, state, self.current_price, reward)
        else:
            # 8 * i + 4 * v + 2 * d + m
            self.record['click'].append(is_clicked)
            self.record['currently_insured'].append(state // 8)
            self.record['number_of_vehicles'].append((state // 4) % 2)
            self.record['number_of_drivers'].append((state // 2) % 2)
            self.record['rank'].append(rank)
            self.record['policies sold'].append(is_sold)
            self.record['married'].append(state % 2)
            self.record['current_price'].append(self.current_price + self.min_price)
            
        self.num_total_runs += 1
        self.num_total_sold += is_sold
        self.num_clicked = (self.num_clicked + is_clicked) % self.num_total
        self.num_sold = (self.num_sold + is_sold) % self.num_total
        

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
        self.n_total_rounds = n_total_rounds
        self.p_vehicle = p_vehicle
        self.p_driver = p_driver
        self.p_insured = p_insured
        self.p_marital = p_marital
        self.n_companies = n_companies
        self.auction_result = []
        self.recording = recording

        # the list of all companies, each company is represented by its index,
        # i.e. self.companies[0] represents Company 0
        self.companies = [Company(updateQ) for _ in range(n_companies)]
        
        self.customer_click_prob = [
            [0.5714285714285714, 0.2, 0.1, 0.0, 0.017324745211109452],
            [0.5095541401273885, 0.2153846153846154, 0.14767932489451474, 0.03821656050955414, 0.01594486133680964],
            [0.5141843971631206, 0.20689655172413796, 0.1946902654867257, 0.04572001049754619, 0.014959805790837303],
            [0.4887459807073955, 0.16149068322981364, 0.08759124087591241, 0.06012767945601441, 0.01580683776918321],
            [0.5553803429645067, 0.19523986521908968, 0.047619047619047616, 0.12121212121212123, 0.017543859649122806],
            [0.5589188420090472, 0.16878783207961265, 0.16756756756756758, 0.031545741324921134, 0.024777006937561945],
            [0.5301685507667087, 0.17525773195876287, 0.125, 0.031007751937984492, 0.010204081632653059],
            [0.5420767126064493, 0.1935483870967742, 0.12307692307692307, 0.03937007874015748, 0.02298850574712644],
            [0.6363636363636364, 0.25, 0.0, 0.07142857142857142, 0.019257668051969424],
            [0.5, 0.13725490196078433, 0.14732142857142858, 0.05921052631578948, 0.011727362001360226],
            [0.5233644859813084, 0.18994413407821228, 0.1366120218579235, 0.05084745762711865, 0.01853420944357649],
            [0.5825242718446602, 0.2085889570552147, 0.1855670103092784, 0.037313432835820885, 0.016744904348269158],
            [0.5592693623101355, 0.19521140912754428, 0.29411764705882354, 0.08, 0.0],
            [0.5615446987596936, 0.18562769777417254, 0.16296296296296298, 0.046052631578947366, 0.016299137104506232],
            [0.5332956548718327, 0.13793103448275862, 0.14393939393939395, 0.036764705882352935, 0.02083333333333333],
            [0.5152471165247118, 0.2318840579710145, 0.16793893129770993, 0.03378378378378378, 0.03333333333333333]
        ]
        self.customer_buy_prob = [
            [0.8974358974358975, 0.05128205128205128, 0.05128205128205128, 0, 0, 0], 
            [0.8205128205128205, 0.10256410256410256, 0.05128205128205128, 0, 0.02564102564102564, 0], 
            [0.8820375335120644, 0.06032171581769437, 0.030831099195710455, 0.020107238605898123, 0.006702412868632708, 0], 
            [0.9128065395095368, 0.04632152588555858, 0.01226158038147139, 0.02316076294277929, 0.005449591280653951, 0], 
            [0.8154657293497364, 0.1335676625659051, 0.033391915641476276, 0.01757469244288225, 0, 0], 
            [0.9471890971039182, 0.027257240204429302, 0.015332197614991482, 0.008517887563884156, 0.0017035775127768314, 0], 
            [0.8620689655172413, 0.11494252873563218, 0.019704433497536946, 0.003284072249589491, 0, 0], 
            [0.9511784511784511, 0.021885521885521887, 0.010101010101010102, 0.015151515151515152, 0.0016835016835016834, 0], 
            [0.9819819819819819, 0, 0, 0.009009009009009009, 0.009009009009009009, 0], 
            [0.978494623655914, 0, 0, 0.021505376344086023, 0, 0], 
            [0.9746646795827124, 0, 0, 0.014903129657228018, 0.003974167908594138, 0.006458022851465474], 
            [0.9844357976653697, 0, 0, 0.00632295719844358, 0.0048638132295719845, 0.004377431906614786], 
            [0.9669421487603306, 0, 0.014462809917355372, 0.012396694214876033, 0.004132231404958678, 0.002066115702479339], 
            [0.9800443458980044, 0, 0.0066518847006651885, 0.008869179600886918, 0.004434589800443459, 0], 
            [0.9725400457665904, 0, 0.016018306636155607, 0.011441647597254004, 0, 0], 
            [0.9794520547945206, 0, 0.0045662100456621, 0.0091324200913242, 0.0045662100456621, 0.00228310502283105]
        ]

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
            The formula is  8 * i + 4 * v + 2 * d + m
        """
        v = np.random.binomial(1, p_vehicle)
        d = np.random.binomial(1, p_driver)
        m = np.random.binomial(1, p_marital)
        i = np.random.binomial(1, p_insured)
        return 8 * i + 4 * v + 2 * d + m

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
        sold = np.random.choice(np.arange(-1, 5), p=self.customer_buy_prob[state])
        
        clicked = []
        not_clicked = []
        for i in range(self.n_companies):
            if i != sold:
                t = np.random.binomial(1, self.customer_click_prob[state][ranking[i]])
                if t == 1:
                    clicked.append(i)
                else:
                    not_clicked.append(i)
        
        return sold, clicked, not_clicked

    def train(self):
        for company in self.companies:
            company.train = True
        for _ in range(self.n_total_rounds):
            self.auction()
    
    def reset(self):
        self.auction_result = []
        for company in self.companies:
            company.reset()
        
    def test(self):
        self.reset()
        for company in self.companies:
            company.train = False
        for _ in range(self.n_total_rounds):
            self.auction()
        
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

        return sorted(indexes, key=price.__getitem__, reverse=True)
    
    def summary(self):
        if self.recording:
            total_sold = [0] * self.n_companies
            for item in self.auction_result:
                total_sold[item['sold']] += 1
            print(total_sold)
