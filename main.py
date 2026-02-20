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

print(gen_table2(db))
