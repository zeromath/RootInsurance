import numpy as np
import pandas as pd
from random import randint

def updateQ(prev_q, next_q, reward, alpha=0.2, discount_factor=0.2):
    return (1 - alpha) * prev_q + alpha * (reward + discount_factor * next_max_q)

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
        self.current_price = 0 # bidding price for current customer
        self.updateQ = updateQ

        # shape: num_clicked  x  num_sold  x  num_states  x  num_price(actions)
        self.q_table = np.zeros([num_total, num_total, num_states, self.max_price - self.min_price + 1])

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
        prev_q = q_table[self.num_clicked, self.num_sold, state, self.current_price]
        next_max_q = np.max(q_table[self.num_clicked + is_clicked, self.num_sold + is_sold])
        q_table[self.num_clicked, self.num_sold, state, self.current_price] = self.updateQ(pewv_q, next_max_q, reward, alpha=0.2, discount_factor=0.2)

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
        reward = 10 # some formula involing current_price and rank
        self.setQ(is_clicked, is_sold, state, self.current_price, reward)
        slef.num_clicked += is_clicked
        self.num_sold += is_sold

    def getPrice(self, state):
        """generate price for this bid

        Parameters:
            state (int): 0-15, current customer's state

        Returns:
            int: the price for this customer
        """
        self.current_price = np.argmax(q_table[self.num_clicked, self.num_sold, state]) + self.min_price
        return self.current_price

class SearchWebsite:
    '''The class of the searching website'''

    def __init__(self, customer_distribution, recording=False):
        """Initialization function

        Parameters:
            customer_distribution (np.array): distribution of customers.
                i-th entry = probability that a customer is in the i-th category
            recording (boolean): whether to record every impression result.

        Returns:
            None
        """
        self.distribution = customer_distribution
        self.recording = recording


    def generateRandomCustomer(self):
        """Generate a random customer

        Parameters:
            self

        Returns:
            int: an int between 0 and 15 representing a customer's state.
        """
        return np.random.choice(16, p=self.distribution)


    def rankBid(self, bids):
        ''' Computes the rank given bids

        Parameters:
            bids (list): list of bids

        Returns:
            rank (list): rank of bids. rank[i] = ranking of i-th bid in bids
        '''
        bids_sorted = sorted( [(bid,company) for company,bid in enumerate(bids)] , reverse=True)
        rank = [0]*len(bids)
        for r, pair in enumerate(bids_sorted):
            rank[pair[1]] = r
        return rank


    def getReward(self, revenue, bid, rank):
        ''' Generates rewards and selling status according to rank
        Args:
            bid (float): our bid
            rank (float): our rank
        Returns:
            rewards (float): rewards
            click (int): 1 iff clicked
            sold (int): 1 iff sold
        '''
        prob_clicking = self.param_['click_prob_'][rank]
        prob_selling = self.param_['sale_prob_'][rank]
        outcome = np.random.rand()
        if outcome < prob_selling:
            # policy sold!
            return (revenue-bid, 1,1)
        if outcome < prob_clicking:
            # clicked but not sold
            return (-bid, 1,0)
        # did not click
        return (0,0,0)











        
################# Zhan's original code #################
    def __init__(self, n_total_rounds, p_vehicle, p_driver, p_insured, p_marital, n_companies = 5, recording = False):
        """Initialization function

        Parameters:
            n_total_rounds (int): number of total rounds
            p_vehicle (float): probability that a customer has two cars
            p_driver (float): probability that a customer has two drivers
            p_insured (float): probability that a customer is insured
            p_marital (float): probability that a customer is married
            n_companys (int): number of companies
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
        self.companies = [Company(n_total_rounds) for _ in range(n_companies)]

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
        return ranking[0], ranking[:3], ranking[3:]

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
        if sold:
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
        # need some way to deal with the same price
        return sorted(range(self.n_companys), key = price.__getitem__)