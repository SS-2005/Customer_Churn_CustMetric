from IPython import get_ipython
from IPython.display import display
# %%
!pip install Faker

import pandas as pd
import numpy as np
from faker import Faker

faker_mumbai = Faker('en_IN')
faker_delhi  = Faker('en_IN')

numrecords = 2000
companytypes = ['Tech', 'Retail', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'E-Commerce', 'Consulting']
marketing_channels = ['Facebook Ads', 'Google Ads', 'Instagram Ads', 'Email Campaign', 'Referral Program', 'Partner Website', 'Word of Mouth', 'Direct Traffic'] 
payment_methods = ['Credit Card', 'Bank Transfer', 'UPI']
np.random.seed(42)

data_delhi = {
    'customerid': [faker_delhi.uuid4() for _ in range(numrecords)],
    'acquisitiondate': [faker_delhi.date_between(start_date='-2y', end_date='today') for _ in range(numrecords)],
    'age': [faker_delhi.random_int(min=18, max=65) for _ in range(numrecords)],
    'gender': [faker_delhi.random_element(elements=('Male', 'Female')) for _ in range(numrecords)],
    'churn': np.random.randint(0, 2, size=numrecords),
    'retentionperiod': np.random.randint(30, 730, size=numrecords),
    'companytype': np.random.choice(companytypes, numrecords),
    'acquisitioncost': [f"₹{np.random.randint(500, 50000):,}" for _ in range(numrecords)],
    'totalspend': [f"₹{np.random.randint(1000, 100000):,}" for _ in range(numrecords)],
    'marketingchannel': np.random.choice(marketing_channels, numrecords),
    'customerinteractions': np.random.randint(1, 50, size=numrecords), 
    'paymentmethod': np.random.choice(payment_methods, numrecords),
    'billingissues': np.random.choice(['Yes', 'No'], numrecords),
    'frequencyofbillingissues': np.where(np.random.choice([True, False], numrecords), 
                                        np.random.randint(1, 10, size=numrecords), 
                                        0) 
}

data_mumbai = {
    'customerid': [faker_mumbai.uuid4() for _ in range(numrecords)],
    'acquisitiondate': [faker_mumbai.date_between(start_date='-2y', end_date='today') for _ in range(numrecords)],
    'age': [faker_mumbai.random_int(min=18, max=65) for _ in range(numrecords)],
    'gender': [faker_mumbai.random_element(elements=('Male', 'Female')) for _ in range(numrecords)],
    'churn': np.random.randint(0, 2, size=numrecords),
    'retentionperiod': np.random.randint(30, 730, size=numrecords),
    'companytype': np.random.choice(companytypes, numrecords),
    'acquisitioncost': [f"₹{np.random.randint(500, 50000):,}" for _ in range(numrecords)],
    'totalspend': [f"₹{np.random.randint(1000, 100000):,}" for _ in range(numrecords)],
    'marketingchannel': np.random.choice(marketing_channels, numrecords), 
    'customerinteractions': np.random.randint(1, 50, size=numrecords),
    'paymentmethod': np.random.choice(payment_methods, numrecords),
    'billingissues': np.random.choice(['Yes', 'No'], numrecords),
    'frequencyofbillingissues': np.where(np.random.choice([True, False], numrecords), 
                                        np.random.randint(1, 10, size=numrecords), 
                                        0)
}

df_mumbai = pd.DataFrame(data_mumbai)
df_delhi = pd.DataFrame(data_delhi)

df_mumbai['city'] = 'Mumbai'
df_delhi['city'] = 'Delhi'

df_mumbai.to_csv('customer_data_Mumbai.csv', index=False)
df_delhi.to_csv('customer_data_Delhi.csv', index=False)

print("Synthetic datasets created and saved as 'customer_data_Mumbai.csv' and 'customer_data_Delhi.csv'.")