import numpy as np
# import pandas as pd
# from random import randint

# def updateQ(prev_q, next_max_q, reward,
#             learning_rate=0.2, discount_factor=0.2):
#     """ Method for updating q_table

#     Parameters:
#         prev_q (float): previous q_value
#         next_max_q (float): maximum expected q_value
#         reward (float): reward given by the environment
#         learning_rate (float): balance between new and old info
#         discount_factor (float): balance between near and far future rewards

#     Returns: the value to be filled into new q_table
#     """
#     return (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q)


#####################################################################
########################### Company class ###########################
#####################################################################

class InsuranceCompany: 

    def __init__(self): 
        self.param_ = dict()
        # Let the bid be uniformly in mean-span, mean+span
        self.param_['mean'] = 10 
        self.param_['span'] = 2
    
    def set_param(self, **kwargs):
        # Use this function to set parameters like married, num_of_vehicles etc 
        for k,v in kwargs.items(): 
            self.param_[k] = v 

    def get_param(self): 
        return self.param_

    def bid(self): 
        bid_mean = self.param_['mean']
        bid_span = self.param_['span']
        return (2*bid_span)*np.random.rand() + bid_mean - bid_span

class Company:
    def __init__(self, choices, distribution):

        self.choices = choices
        self.distribution = distribution

    def bid(self, customer_type):
        """ Bid according to distribution
        """
        return \
        np.random.choice(self.choices, p=self.distribution[customer_type, :])

# class Company:
#     '''The class of all companies'''
#     def __init__(self, num_rounds, choices, q_table=None, num_states=16):
#         self.num_rounds = num_rounds
#         self.choices = choices
#         self.customer_type = None
#
#         # q_table is a constant vector defining the distribution of bids for
#         # combination of state. Each row i sum up to 1, representing the
#         # distribution of bids for the i-th type of customer
#         if q_table is not None:
#             self.q_table = q_table
#         else:
#             self.q_table = np.zeros((num_states, choices.size))
#
#     def set_q_table(self, table):
#         """ Set q_table """
#         self.q_table = table
#
#
#     def setCustomerType(self, state):
#         """ Update current state after receiving the information given by the Bidding Website
#
#         Parameters:
#             state (int): The customer type as given by the Bidding Website
#
#         Returns:
#             None
#         """
#         self.customer_type = state
#
#     def bid(self):
#         """ Bid according to distribution
#         """
#         # print(self.choices)
#         # # print(self.q_table)
#         return \
#         np.random.choice(self.choices,p=self.q_table[self.customer_type, :])
#
#         # return self.q_table[self.customer_type, choice]


#####################################################################
######################## Smarrt Company class #######################
#####################################################################
#
# class SmartCompany:
#     '''The class of a smart companies that uses RL to optimize its strategies.
#     '''
#
#     def __init__(self, num_rounds, choices, q_table, learning_rate=.2,
#                  discount_factor=.9, num_states=16, epsilon=.2):
#         """ Initialization function
#
#         Parameters:
#             num_rounds (int):   Number of total rounds
#             choices (np.array): Possible choices for bid price
#             num_states (int):   Default 16 states (8 * v + 4 * d + 2 * m + i)
#
#         Returns:
#             None
#         """
#         self.num_rounds = num_rounds
#         self.num_clicked = 0
#         self.num_sold = 0
#         self.choices = choices
#
#         self.last_choice = None
#         self.current_bid = None
#         self.current_action_index = None
#
#         self.customer_type = None # state S_t
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.epsilon = epsilon
#
#         # Initialize q table
#         if q_table is not None:
#             self.q_table = q_table
#         else:
#             self.q_table = np.random.rand(num_rounds, num_rounds,
#                                           num_states, choices.size)
#     def reset(self):
#         """ Resetting the company for a new episode but retain information from
#             before
#         """
#         self.num_clicked = 0
#         self.num_sold = 0
#
#         self.last_choice = None
#         self.current_bid = None
#         self.current_action_index = None
#
#         self.customer_type = None # state S_t
#
#
#
#     def computeReward(self, is_clicked, is_sale, bid, revenue=20):
#         """ Compute reward for our action, given the clicking and sale
#             information returned by the Bidding Website
#
#         Parameters:
#             clicked (int): 1 if clicked, 0 if not clicked
#             sale (int):    1 if policy sold, 0 if not sold
#         """
#         return -bid*is_clicked + revenue*is_sale
#
#     def setCustomerType(self, state):
#         """ Update current state after receiving the information given by the Bidding Website
#
#         Parameters:
#             state (int): The customer type as given by the Bidding Website
#
#         Returns:
#             None
#         """
#         self.customer_type = state
#
#
#     def bid(self):
#         """generate price for this bid
#
#         Parameters:
#             None
#
#         Returns:
#             int: the bid we submit
#         """
#
#         # Get the index of optimal choice,
#         # i.e. choice that maxize expected payoff
#         should_explore = np.random.rand()
#         if should_explore < self.epsilon:
#             # Randomly explore
#             index_of_optimal_choice = np.random.randint(self.choices.size)
#         else:
#             # Choose optimal strategy
#             index_of_optimal_choice = \
#                 np.argmax(self.q_table[self.num_clicked, self.num_sold,
#                                        self.customer_type, :])
#
#
#         # Get choice
#         self.current_action_index = index_of_optimal_choice
#         self.current_bid = self.choices[index_of_optimal_choice]
#
#         # Return optimal choice
#         return self.current_bid
#
#
#     # def getMaxQ(self, new_customer, num_clicked, num_sold):
#     #     """ Get max_{a'} Q(s',a')
#     #     """
#     #     return np.max(self.q_table[num_clicked, num_sold, new_customer])
#
#     def updateQ(self, old_customer, new_customer, action_index, reward):
#         """ Update Q table. This is triggered when we have made a new choice
#             and a new state
#
#         Parameters:
#             is_clicked (int):        0 for not clicked, 1 for clicked
#             is_sold (int):           0 for not clicked, 1 for clicked
#             state (int):             0-15, current customer's state
#             action_index(int):       choice among possible actions
#             reward (int):            the reward resulting from the bidding price
#             alpha (float):           0-1, learning_rate
#             discount_factor (float): 0-1, factor discounting future rewards
#
#         Returns:
#             None
#         """
#         # Get previous q_table[state, action]
#         prev_q = self.q_table[self.num_clicked, self.num_sold,
#                               old_customer, action_index]
#
#         # Retrieve previous optimal future value
#         optimal_future_value = np.max(self.q_table[self.num_clicked,
#                                                    self.num_sold,
#                                                    new_customer, :])
#
#         # Update q_table according to rewards received
#         self.q_table[self.num_clicked, self.num_sold,
#                      old_customer, action_index] = \
#             updateQ(prev_q, optimal_future_value,
#                     reward, self.learning_rate, self.discount_factor)
#
#
#     def updateState(self, is_clicked, is_sold, rank, new_customer_type):
#         """update q_table, num_sold, num_clicked; calculating reward
#
#         Parameters:
#             is_clicked (int): 0 for not clicked, 1 for clicked
#             is_sold (int):    0 for not clicked, 1 for clicked
#             rank (int):       the rank of this bid.
#             new_customer_type (int):  0-15, current customer's state
#
#         Returns:
#             None
#         """
#         # Compute reward to update q_table
#         reward = self.computeReward(is_clicked, is_sold, self.current_bid)
#
#         self.updateQ(self.customer_type, new_customer_type,
#                      self.current_action_index, reward)
#
#         self.num_clicked += is_clicked
#         self.num_sold += is_sold

#######################################################################
######################### SearchWebsite class #########################
#######################################################################

class SearchWebsite:
    '''The class of bidding environment'''

    def __init__(self, p_click, p_sale):
        """Initialization function

        Parameters:
            None

        Returns:
            None
        """

        self.distribution = np.array([0.0039, 0.0746, 0.0569, 0.0609, 0.0111, 0.2013, 0.0484, 0.0437,
       0.0039, 0.0734, 0.0587, 0.0594, 0.0093, 0.2056, 0.0451, 0.0438])

        self.p_click = p_click

        self.p_sale = p_sale


    def generateRandomCustomer(self):
        """Generate a random customer

        Parameters:
            self

        Returns:
            int: an int between 0 and 15 representing a customer's state.
        """
        return np.random.choice(self.distribution.size, p=self.distribution)


    def rankBid(self, bids):
        ''' Computes the rank given bids

        Parameters:
            bids (list): list of bids

        Returns:
            rank (list): rank of bids. rank[i] = ranking of i-th company in bids
        '''
        bids_sorted = sorted( [(bid,company) for company,bid in enumerate(bids)] , reverse=True)
        rank = [0]*len(bids)
        for r, pair in enumerate(bids_sorted):
            rank[pair[1]] = r
        return rank


    def rewardDecision(self, customer_type, rank):
        ''' Generates rewards and selling status according to rank

        Parameters:
            customer_type (int): customer type
            rank (int): the company's rank

        Returns:
            click (int): 1 iff clicked
            sold (int): 1 iff sold
        '''
        prob_clicking = self.p_click[customer_type, rank]
        prob_selling = self.p_sale[customer_type, rank]
        outcome = np.random.rand()
        if (outcome < prob_selling).all():
            # policy sold!
            return (1,1)
        if (outcome < prob_clicking).all():
            # clicked but not sold
            return (1,0)
        # did not click
        return (0,0)
