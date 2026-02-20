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
    return table1

#generate table1 and export to excel
table1 = gen_table1(db)
table1.to_excel("Table1.xlsx")
