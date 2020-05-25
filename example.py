import numpy as np
# import sys, os
# sys.path.insert(0, os.getcwd())
import roles_simple as rs

def normalize_random(num_rows, num_cols):
    ans = np.random.rand(num_rows, num_cols)
    ans = ans/ans.sum(axis=1)[:,None]
    return ans

######################################################
#################### Sample calls ####################
######################################################

# Define global constants
NUM_TYPE_CUSTOMERS = 16
NUM_COMPETITORS = 4
NUM_ROUNDS = 20
CHOICES = np.array(np.linspace(9,12,4))
NUM_EPISODES = 100

### Initialize objects

# Initialize competitors with random strategy
competitors_list = [None]*NUM_COMPETITORS
for i in range(NUM_COMPETITORS):
    competitor = rs.Company(NUM_ROUNDS, CHOICES, None)
    competitor.set_q_table(normalize_random(NUM_TYPE_CUSTOMERS,
                                            competitor.choices.size))
    competitors_list[i] = competitor


# Initialize our company
us = rs.SmartCompany(NUM_ROUNDS, CHOICES, None)
companies_list = [us] + competitors_list

# Initialize Website
customer_dist = np.random.rand(NUM_TYPE_CUSTOMERS)
customer_dist = customer_dist/customer_dist.sum()

prob_click = np.random.rand(16,5)
prob_sale = np.multiply(np.random.rand(16,5), prob_click)

website = rs.SearchWebsite(customer_dist, prob_click, prob_sale)


### One episode
print("Initial q_table")
print(us.q_table[10, 10, 1, :])
for episode in range(NUM_EPISODES):
    # Redefine the num_clicked in our company
    us.reset()

    # Initialize first customer type
    customer_type = website.generateRandomCustomer()

    for _ in range(NUM_ROUNDS):

        # Set customer type for all companies
        for c in companies_list:
            c.setCustomerType(customer_type)

        # Ask all companies to bid
        bids = [c.bid() for c in companies_list]

        # Get the rank of our company
        rank = website.rankBid(bids)[0]
        is_clicked, is_sold = website.rewardDecision(customer_type, rank)

        # Get next customer type, to update our q_table
        customer_type = website.generateRandomCustomer()
        us.updateState(is_clicked, is_sold, rank, customer_type)

print("After 10 episodes: q_table is")
print(us.q_table[10, 10, 1, :])
print("""Note that q_table result probably did not change. This is because
         our update process is really sparse, so it takes a lot of episodes
         to get to a particular element in the table.""")
























########################################################
################ Testing Initialization ################
########################################################

### Initialize competitors
# NUM_TYPE_CUSTOMERS = 16
# competitor_choices = np.array(np.linspace(9,12,4))
# # competitor = rs.Company(10, competitor_choices)
# #
# competitor_q_table = normalize_random(NUM_TYPE_CUSTOMERS,
#                                       competitor_choices.size)
# print(competitor_q_table)
#     np.random.rand(NUM_TYPE_CUSTOMERS,
#                    competitor_choices.size)
# competitor_q_table = \
#     competitor_q_table/competitor_q_table.sum(axis=1)[:, None]
# competitor.set_q_table(competitor_q_table)
#
# competitor.setCustomerType(8)


# ### Initialize search website
# customer_dist = \
#     np.random.rand(NUM_TYPE_CUSTOMERS)
# customer_dist = \
#     customer_dist/customer_dist.sum()
#
# prob_click = np.random.rand(16,5) * .5
# prob_sale = np.multiply(np.random.rand(16,5), prob_click)
#
#
# website = rs.SearchWebsite(customer_dist)
# website.set_prob_click(prob_click)
# website.set_prob_sale(prob_sale)

## testing behaviors of website
# print(website.generateRandomCustomer())
# bids = np.random.rand(5)
# print(bids)
# print(website.rankBid(bids))
# print(website.rewardDecision(11,0))

### Initialize Smart Company
# us = rs.SmartCompany(10,competitor_choices, None)
# us.setCustomerType(11)
# print(us.bid())
# print(us.current_action_index)
# print(us.q_table)
