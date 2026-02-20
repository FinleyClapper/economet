"""
@file       main.py
@author     Finley Clapper
@course     Applied Econometrics I (ECON: 325)
@institution The University of Akron
@date       Spring 2026
@description TBD
"""

import os
import pandas as pd

#intialize db
db = pd.read_csv(os.path.join(os.getcwd(), "DB1.csv"))

#gerneate and return table1 (Descriptive stats for continous/discrete variables)
def gen_table1(db: pd.DataFrame) -> pd.DataFrame:
    collumns = ['nchild', 'wkswork', 'age', 'income']
    subset = db[collumns]
    table1 = subset.describe()
    drops = ['25%', '50%', '75%']
    table1 = table1.drop(drops)
    names = {'count': 'N',
             'mean': 'Mean',
             'std': 'Std Dev',
             'min': 'Min',
             'max': 'Max'}
    table1 = table1.rename(index=names)
    table1 = table1.round(2)
    table1 = table1.T
    return table1

#generate table1 and export to excel
table1 = gen_table1(db)
print(table1)
#table1.to_excel("Table1.xlsx")

#gerneate and return table2 (Descriptive stats for categorical variables)
def gen_table2(db: pd.DataFrame) -> pd.DataFrame:
    #Setup labels for all the rows
    labels = {
    'gender': {
        0: 'Female', 
        1: 'Male'
    },
    'raceethnic': {
        1: 'White', 
        2: 'Black', 
        3: 'Asian', 
        4: 'Hispanic', 
        5: 'Other'
    },
    'ed': {
        1: 'No high school diploma',
        2: 'High school graduate',
        3: 'Some college',
        4: 'College graduate',
        5: 'Graduate education'
    },
    'marst': {
        1: 'Married, spouse present',
        2: 'Married, spouse absent',
        3: 'Separated',
        4: 'Divorced',
        5: 'Widowed',
        6: 'Never married/single'
    }
}
    
    #Label categorical variables
    for col, mapping in labels.items():
        if col in db.columns:
            db[col] = db[col].map(mapping).astype('category')

    all_vars = []

    #Get counts and percents for each category
    for col in labels.keys():
        if col in db.columns:
            counts = db[col].value_counts().sort_index()
            percents = (db[col].value_counts(normalize=True).sort_index() * 100).round(2)
            temp_db = pd.DataFrame({
                'Frequency': counts,
                'Percent': percents
            })
            all_vars.append(temp_db)

    table2 = pd.concat(all_vars)               
    return table2

def gen_state_table(db: pd.DataFrame) -> pd.DataFrame:

    #dict to translate codes to state names
    names = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
    27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
    31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
    35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
    39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island',
    45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 48: 'Texas',
    49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia',
    55: 'Wisconsin', 56: 'Wyoming'
}
    #get counts and percents for states
    state_counts = db['statefips'].value_counts()
    state_percents = (db['statefips'].value_counts(normalize=True) * 100).round(2)
    
    #make table with all state counts and percents
    states = pd.DataFrame({
        'Frequency': state_counts,
        'Percent': state_percents
    }).sort_values('Frequency', ascending=False)

    #get top and bottom 5
    top5 = states.head(5)
    bottom5 = states.tail(5)
    
    #create final state table
    state_table = pd.concat([top5, bottom5], keys=['Top 5 States', 'Bottom 5 States'])
    state_table = state_table.reset_index()
    state_table['statefips'] = state_table['statefips'].map(names)
    state_table.columns = ['Group', 'State', 'Frequency', 'Percent']
    return state_table

print(gen_table2(db))
print(gen_state_table(db))