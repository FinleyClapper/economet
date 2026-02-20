###READ ME############################
"""
Note 1: You MUST place the data file(s) in the same folder as your code file.
        The same applies to any output files you want to save.
Note 2: You MUST have the relevant packages installed. In PyCharm, you can identify
        and install missing packages by examining errors in the Run console.
"""
######################################
# Import necessary packages.
import pandas as pd
import os
import statsmodels.formula.api as smf

# Read the data file: See Note 1 in the READ ME section above.
caschool = pd.read_excel(os.path.join(os.getcwd(), "caschool.xlsx"), sheet_name="caschool")

# Quick check to ensure data is read/imported properly.
print(caschool.head(7))