import pandas as pd

def getData():

    acme_raw = pd.read_excel('Acme.xlsx')
    acme = acme_raw.copy()

    acme = acme.drop('cost', axis = 1)

    acme['policies_sold'] = acme['policies sold']
    acme = acme.drop('policies sold', axis = 1)

    acme['married'] = pd.get_dummies(acme['marital_status'])['M']
    acme = acme.drop('marital_status', axis = 1)

    acme['insured'] = pd.get_dummies(acme['currently_insured'])['Y']
    acme = acme.drop('currently_insured', axis = 1)
    return acme, acme_raw