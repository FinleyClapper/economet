"""
@file       main.py
@author     Finley Clapper
@course     Applied Econometrics I (ECON: 325)
@institution The University of Akron
@date       Spring 2026
@description Class Project - Parts 1 & 2
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tabulate import tabulate
import statsmodels.formula.api as smf

#intialize db
db = pd.read_csv(os.path.join(os.getcwd(), "DB1.csv"))

#generate and return table1 (Descriptive stats for continous/discrete variables)
def gen_table1(db: pd.DataFrame) -> pd.DataFrame:

    #get only data we want from collumns we want
    collumns = ['nchild', 'wkswork', 'age', 'income']
    subset = db[collumns]
    table1 = subset.describe()
    drops = ['25%', '50%', '75%']
    table1 = table1.drop(drops)

    #rename collumns and do intial formatings
    names = {'count': 'N',
             'mean': 'Mean',
             'std': 'Std Dev',
             'min': 'Min',
             'max': 'Max'}
    table1 = table1.rename(index=names)
    table1 = table1.round(2)
    table1 = table1.T

    #Format rows/cols
    table1 = table1.astype(object)
    cols = ['Mean', 'Std Dev', 'Min', 'Max']
    table1.loc['income', cols] = table1.loc['income', cols].map(lambda x: f"${x:,.2f}")
    rows = ['nchild', 'wkswork', 'age']
    table1.loc[rows, cols] = table1.loc[rows, cols].map(lambda x: f"{x:,.2f}")
    table1['N'] = table1['N'].map(lambda x: f"{x:,.0f}")
    return table1

#generate table1 and export to excel
table1 = gen_table1(db)
table1.to_excel("Table1.xlsx")

#gerneate and return table2 (Descriptive stats for categorical variables)
def gen_table2(db: pd.DataFrame) -> pd.DataFrame:
    #Setup labels for all the rows
    labels = {
    'male': {
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

    table2['Frequency'] = table2['Frequency'].map(lambda x: f"{x:,}")
    table2['Percent'] = table2['Percent'].map(lambda x: f"{x:.2f}%")      
    return table2

#generate table1 and export to excel
table2 = gen_table2(db)
table2.to_excel("Table2.xlsx")

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

    states.index = states.index.map(names)

    #get top and bottom 5
    top5 = states.head(5)
    bottom5 = states.tail(5)
    
    #create final state table
    state_table = pd.concat([top5, bottom5], keys=['Top 5 States', 'Bottom 5 States'])
    state_table.index.names = [None, None]
    state_table['Frequency'] = state_table['Frequency'].map(lambda x: f"{x:,}")
    state_table['Percent'] = state_table['Percent'].map(lambda x: f"{x:.2f}%")
    return state_table

#generate table1 and export to excel
states = gen_state_table(db)
states.to_excel("States.xlsx")

def gen_income_chart(db: pd.DataFrame):
    #sort by education and get mean for each group
    avg_by_ed = db.groupby('ed')['income'].mean().sort_values(ascending=True)
    
    #setup chart
    plt.figure(figsize=(10, 6))
    x_ticks = range(len(avg_by_ed))
    
    #create the bars
    plt.bar(x_ticks, avg_by_ed.values, width=0.5, color='gray', edgecolor='black')
    
    #Add labels
    plt.xlabel("Education Level", fontsize=12, fontweight='bold')
    plt.ylabel("Average Income ($)", fontsize=12, fontweight='bold')
    plt.title('Figure 1. Average Income by Educational Attainment', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, avg_by_ed.index, rotation=15)
    for i, value in enumerate(avg_by_ed):
        plt.text(i, value, f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

    #saves the chart
    plt.tight_layout()
    plt.savefig("Income_by_Education_Chart.jpg", dpi=300)
    plt.show()
gen_income_chart(db)

def gen_gender_gap_table(db: pd.DataFrame) -> pd.DataFrame:
    # define races
    races = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    
    t_table_rows = []
    
    for race_name in races:
        # filter race data
        race_data = db[db['raceethnic'] == race_name]
        
        # filter male/female inc
        men_inc = race_data[race_data['male'] == 'Male']['income']
        women_inc = race_data[race_data['male'] == 'Female']['income']
        
        # fix the small sample error
        if len(men_inc) > 1 and len(women_inc) > 1:
            mean_m = men_inc.mean()
            mean_w = women_inc.mean()
            diff = mean_m - mean_w
            
            # t-test
            p_val = ttest_ind(men_inc, women_inc, nan_policy='omit', equal_var=False).pvalue
            
            #add stars
            stars = ""
            if p_val <= 0.01: stars = "***"
            elif p_val <= 0.05: stars = "**"
            elif p_val <= 0.10: stars = "*"
            
            t_table_rows.append([
                race_name,
                f"${mean_m:,.2f}",
                f"${mean_w:,.2f}",
                f"${diff:,.2f}{stars}"
            ])
        else:
            #incase theres no data
            t_table_rows.append([race_name, "No Data", "No Data", "No Data"])

    columns = ['Race', 'Men Avg. Income', 'Women Avg. Income', 'Difference (Men-Women)']
    return pd.DataFrame(t_table_rows, columns=columns)

# Generate and Print
gender_gap_table = gen_gender_gap_table(db)
gender_gap_table.to_excel("Gender.xlsx")

# ===========================================================
# PART 2: IDENTIFY DETERMINANTS OF INCOME
# ===========================================================

# Load a fresh copy of DB1 (original numeric codes, unmodified by Part 1)
db2 = pd.read_csv(os.path.join(os.getcwd(), "DB1.csv"))


# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def find_file(filename):
    """Search current directory and up to 5 parent directories for a file."""
    path = os.getcwd()
    for _ in range(6):
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate
        path = os.path.dirname(path)
    return os.path.join(os.getcwd(), filename)


def run_f_test(model, param_list):
    """Joint F-test: are all parameters in param_list jointly zero?"""
    params = [p for p in param_list if p in model.params.index]
    if not params:
        return None, 1.0
    all_p = list(model.params.index)
    R = np.zeros((len(params), len(all_p)))
    for j, p in enumerate(params):
        R[j, all_p.index(p)] = 1
    res = model.f_test(R)
    return float(res.fvalue), float(res.pvalue)


def main_cat_params(model, var):
    """Parameter names for the main-effect dummies of a categorical variable (no interactions)."""
    return [p for p in model.params.index if f'C({var}' in p and ':' not in p]


def fmt_cell(coef, se, pval):
    """Format coefficient with significance stars and SE string."""
    stars = '***' if pval <= 0.01 else '**' if pval <= 0.05 else '*' if pval <= 0.10 else ''
    return f"{coef:,.2f}{stars}", f"({se:,.2f})"


def get_cell(model, param):
    """Get (coef_string, se_string) for a parameter; ('', '') if absent."""
    if param not in model.params.index:
        return '', ''
    return fmt_cell(model.params[param], model.bse[param], model.pvalues[param])


# -------------------------------------------------------
# MODEL 1: Baseline (no state, occupation, or industry)
# -------------------------------------------------------
model1 = smf.ols(
    "income ~ male + C(raceethnic, Treatment(1)) + C(ed, Treatment(1)) + "
    "C(marst, Treatment(1)) + nchild + wkswork + age",
    data=db2
).fit(cov_type='HC1')

# -------------------------------------------------------
# MODEL 2: Add State, Occupation, and Industry Controls
# -------------------------------------------------------
model2 = smf.ols(
    "income ~ male + C(raceethnic, Treatment(1)) + C(ed, Treatment(1)) + "
    "C(marst, Treatment(1)) + nchild + wkswork + age + "
    "C(statefips, Treatment(1)) + C(occ, Treatment(1)) + C(ind, Treatment(1))",
    data=db2
).fit(cov_type='HC1')

# -------------------------------------------------------
# MODEL 3: Drop variables insignificant at 5% from Model 2
# -------------------------------------------------------
# Continuous/discrete: individual t-test (p-value)
# Categorical groups:  joint F-test

cont_vars_m2 = ['male', 'nchild', 'wkswork', 'age']
cat_vars_m2  = ['raceethnic', 'ed', 'marst', 'statefips', 'occ', 'ind']

sig_cont = [v for v in cont_vars_m2 if model2.pvalues.get(v, 1.0) < 0.05]
sig_cat  = [v for v in cat_vars_m2
            if run_f_test(model2, main_cat_params(model2, v))[1] < 0.05]

m3_terms = sig_cont + [f"C({v}, Treatment(1))" for v in sig_cat]
model3 = smf.ols("income ~ " + " + ".join(m3_terms), data=db2).fit(cov_type='HC1')

# Track which geographic/occupational controls survived into Model 3
m3_has_state = 'statefips' in sig_cat
m3_has_occ   = 'occ'       in sig_cat
m3_has_ind   = 'ind'       in sig_cat

# -------------------------------------------------------
# MODEL 4: Add Interaction Terms to Model 3
# -------------------------------------------------------
# Q1: Does the gender gap in income depend on education level?
# Q2: Does the gender gap in income depend on race?
# Q3: Does the return to education depend on race?

# Ensure all main effects needed for interactions are present
m4_main = m3_terms.copy()
for needed in ['male', 'C(raceethnic, Treatment(1))', 'C(ed, Treatment(1))']:
    if needed not in m4_main:
        m4_main.append(needed)

m4_terms = m4_main + [
    'male:C(ed, Treatment(1))',                              # Q1
    'male:C(raceethnic, Treatment(1))',                      # Q2
    'C(ed, Treatment(1)):C(raceethnic, Treatment(1))'        # Q3
]
model4 = smf.ols("income ~ " + " + ".join(m4_terms), data=db2).fit(cov_type='HC1')

# Joint F-tests for each interaction set (reported in write-up)
m_ed_params    = [p for p in model4.params.index if 'male' in p and 'C(ed' in p]
m_race_params  = [p for p in model4.params.index if 'male' in p and 'C(raceethnic' in p]
ed_race_params = [p for p in model4.params.index if 'C(ed' in p and 'C(raceethnic' in p]

fstat_q1, pval_q1 = run_f_test(model4, m_ed_params)
fstat_q2, pval_q2 = run_f_test(model4, m_race_params)
fstat_q3, pval_q3 = run_f_test(model4, ed_race_params)

print("\n--- Model 4 Interaction F-Tests ---")
print(f"Q1 (Male x Education):  F={fstat_q1:.4f}, p={pval_q1:.4f}")
print(f"Q2 (Male x Race):       F={fstat_q2:.4f}, p={pval_q2:.4f}")
print(f"Q3 (Education x Race):  F={fstat_q3:.4f}, p={pval_q3:.4f}")

# -------------------------------------------------------
# MODEL 5: Drop insignificant variables from Model 4 at 5%
# (interaction sets tested separately from main effects)
# -------------------------------------------------------
m5_terms = []

# Re-test main effects using Model 4 estimates
for term in m4_main:
    if term.startswith('C('):
        var = term.split('(')[1].split(',')[0]
        _, p = run_f_test(model4, main_cat_params(model4, var))
        if p < 0.05:
            m5_terms.append(term)
    else:
        if model4.pvalues.get(term, 1.0) < 0.05:
            m5_terms.append(term)

# Keep interaction sets that are jointly significant
def _ensure(terms, needed):
    for n in needed:
        if n not in terms:
            terms.append(n)

if pval_q1 < 0.05:
    _ensure(m5_terms, ['male', 'C(ed, Treatment(1))'])
    m5_terms.append('male:C(ed, Treatment(1))')

if pval_q2 < 0.05:
    _ensure(m5_terms, ['male', 'C(raceethnic, Treatment(1))'])
    m5_terms.append('male:C(raceethnic, Treatment(1))')

if pval_q3 < 0.05:
    _ensure(m5_terms, ['C(ed, Treatment(1))', 'C(raceethnic, Treatment(1))'])
    m5_terms.append('C(ed, Treatment(1)):C(raceethnic, Treatment(1))')

model5 = smf.ols("income ~ " + " + ".join(m5_terms), data=db2).fit(cov_type='HC1')

m5_has_state = 'C(statefips, Treatment(1))' in m5_terms
m5_has_occ   = 'C(occ, Treatment(1))'       in m5_terms
m5_has_ind   = 'C(ind, Treatment(1))'       in m5_terms

# -------------------------------------------------------
# MODEL 6: IPUMS ACS - Pennsylvania 2023
# -------------------------------------------------------

def map_race_ipums(race, hispan):
    """Map IPUMS RACE + HISPAN to 5-category raceethnic."""
    if hispan > 0:   return 4  # Hispanic
    if race == 1:    return 1  # White
    if race == 2:    return 2  # Black
    if race in [4, 5, 6]: return 3  # Asian / Pacific Islander
    return 5  # Other (American Indian, mixed, etc.)


def map_ed_ipums(educ):
    """Map IPUMS EDUC (0–11) to 5-category ed."""
    if educ <= 5:  return 1  # No HS diploma
    if educ == 6:  return 2  # HS graduate
    if educ <= 9:  return 3  # Some college
    if educ == 10: return 4  # 4-year college graduate
    return 5                 # Graduate education (5+ years)


def map_occ_ipums(occ):
    """Map IPUMS detailed OCC codes to 16 consolidated categories."""
    if occ == 0:              return 0   # N/A
    if 10   <= occ <= 430:    return 1   # Management
    if 500  <= occ <= 740:    return 2   # Business & Finance
    if 800  <= occ <= 1240:   return 3   # Computer, Math & Engineering
    if 1300 <= occ <= 1560:   return 4   # Life, Physical & Social Science
    if 1600 <= occ <= 1960:   return 5   # Community, Social Service & Legal
    if 2000 <= occ <= 2060:   return 6   # Education
    if 2100 <= occ <= 2180:   return 7   # Arts, Entertainment & Media
    if 2200 <= occ <= 2550:   return 8   # Healthcare Practitioners
    if 2600 <= occ <= 2970:   return 9   # Healthcare Support
    if 3000 <= occ <= 3550:   return 10  # Protective Service
    if 3600 <= occ <= 3960:   return 11  # Food Prep, Cleaning & Personal Care
    if 4000 <= occ <= 4660:   return 12  # Sales, Office & Admin Support
    if 4700 <= occ <= 5940:   return 13  # Farming, Construction & Mining
    if 6000 <= occ <= 7620:   return 14  # Production, Transport & Repair
    if 9800 <= occ <= 9830:   return 15  # Military
    if occ == 9920:           return 16  # Unemployed / last job
    return 0


def map_ind_ipums(ind):
    """Map IPUMS detailed IND codes to 15 consolidated categories."""
    if ind == 0:               return 0   # N/A
    if 170  <= ind <= 290:     return 1   # Agriculture, Forestry & Fishing
    if 370  <= ind <= 490:     return 2   # Mining
    if ind == 570:             return 3   # Construction
    if 1070 <= ind <= 3990:    return 4   # Manufacturing
    if 4070 <= ind <= 4590:    return 5   # Wholesale Trade
    if 4670 <= ind <= 5790:    return 6   # Retail Trade
    if 6070 <= ind <= 6390:    return 7   # Transportation & Warehousing
    if 6470 <= ind <= 6780:    return 8   # Information
    if 6870 <= ind <= 7190:    return 9   # Finance & Insurance
    if 7270 <= ind <= 7790:    return 10  # Professional & Real Estate Services
    if 7860 <= ind <= 8470:    return 11  # Educational Services
    if 8560 <= ind <= 8690:    return 12  # Healthcare & Social Assistance
    if 8770 <= ind <= 9290:    return 13  # Arts, Entertainment & Food Services
    if 9370 <= ind <= 9590:    return 14  # Other Services
    if 9670 <= ind <= 9870:    return 15  # Public Administration
    return 0


# Load and clean IPUMS data
ipums_path = find_file("usa_00002.csv")
ipums_raw = pd.read_csv(ipums_path)

# Filter to Pennsylvania 2023
pa = ipums_raw[(ipums_raw['STATEFIP'] == 42) & (ipums_raw['YEAR'] == 2023)].copy()

# Recode variables to match DB1 structure
pa['male']      = (pa['SEX'] == 1).astype(int)
pa['raceethnic'] = pa.apply(lambda r: map_race_ipums(r['RACE'], r['HISPAN']), axis=1)
pa['ed']        = pa['EDUC'].apply(map_ed_ipums)
pa['marst']     = pa['MARST']
pa['nchild']    = pa['NCHILD']
pa['wkswork']   = pa['WKSWORK1']
pa['age']       = pa['AGE']
pa['income']    = pa['INCWAGE']
pa['occ']       = pa['OCC'].apply(map_occ_ipums)
pa['ind']       = pa['IND'].apply(map_ind_ipums)

# Keep only workers with positive, non-missing wage income and a valid occupation
pa_clean = pa[
    (pa['income'] > 0) & (pa['income'] < 999998) &
    (pa['wkswork'] > 0) &
    (pa['occ'] > 0) & (pa['ind'] > 0)
].copy()

print(f"\nIPUMS PA 2023 sample: {len(pa_clean):,} observations")
print(f"Average income (INCWAGE): ${pa_clean['income'].mean():,.2f}")

# Model 6: same variables as Model 5 but on PA 2023 IPUMS data, always with
# occ, ind, and all three interaction sets.
model6 = smf.ols(
    "income ~ male + C(raceethnic, Treatment(1)) + C(ed, Treatment(1)) + "
    "C(marst, Treatment(1)) + nchild + wkswork + age + "
    "C(occ, Treatment(1)) + C(ind, Treatment(1)) + "
    "male:C(raceethnic, Treatment(1)) + "
    "male:C(ed, Treatment(1)) + "
    "C(ed, Treatment(1)):C(raceethnic, Treatment(1))",
    data=pa_clean
).fit(cov_type='HC1')


# -------------------------------------------------------
# COMBINED REGRESSION RESULTS TABLE
# -------------------------------------------------------

# Ordered list of (parameter_name, display_label) for the main table
# State/occ/ind dummies are excluded; replaced by control indicator rows below.
PARAM_ROWS = [
    ('Intercept',                                                    'Intercept'),
    ('male',                                                         'Male'),
    ('C(raceethnic, Treatment(1))[T.2]',                            'Black'),
    ('C(raceethnic, Treatment(1))[T.3]',                            'Asian'),
    ('C(raceethnic, Treatment(1))[T.4]',                            'Hispanic'),
    ('C(raceethnic, Treatment(1))[T.5]',                            'Other Race'),
    ('C(ed, Treatment(1))[T.2]',                                    'HS Graduate'),
    ('C(ed, Treatment(1))[T.3]',                                    'Some College'),
    ('C(ed, Treatment(1))[T.4]',                                    'College Graduate'),
    ('C(ed, Treatment(1))[T.5]',                                    'Graduate Education'),
    ('C(marst, Treatment(1))[T.2]',                                 'Married, Spouse Absent'),
    ('C(marst, Treatment(1))[T.3]',                                 'Separated'),
    ('C(marst, Treatment(1))[T.4]',                                 'Divorced'),
    ('C(marst, Treatment(1))[T.5]',                                 'Widowed'),
    ('C(marst, Treatment(1))[T.6]',                                 'Never Married'),
    ('nchild',                                                       'Nchild'),
    ('wkswork',                                                      'Weeks Worked'),
    ('age',                                                          'Age'),
    # Male × Education interactions
    ('male:C(ed, Treatment(1))[T.2]',                               'Male × HS Graduate'),
    ('male:C(ed, Treatment(1))[T.3]',                               'Male × Some College'),
    ('male:C(ed, Treatment(1))[T.4]',                               'Male × College Graduate'),
    ('male:C(ed, Treatment(1))[T.5]',                               'Male × Graduate Education'),
    # Male × Race interactions
    ('male:C(raceethnic, Treatment(1))[T.2]',                       'Male × Black'),
    ('male:C(raceethnic, Treatment(1))[T.3]',                       'Male × Asian'),
    ('male:C(raceethnic, Treatment(1))[T.4]',                       'Male × Hispanic'),
    ('male:C(raceethnic, Treatment(1))[T.5]',                       'Male × Other Race'),
    # Education × Race interactions
    ('C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.2]',  'HS Grad × Black'),
    ('C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.3]',  'HS Grad × Asian'),
    ('C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.4]',  'HS Grad × Hispanic'),
    ('C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.5]',  'HS Grad × Other'),
    ('C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.2]',  'Some College × Black'),
    ('C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.3]',  'Some College × Asian'),
    ('C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.4]',  'Some College × Hispanic'),
    ('C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.5]',  'Some College × Other'),
    ('C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.2]',  'College Grad × Black'),
    ('C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.3]',  'College Grad × Asian'),
    ('C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.4]',  'College Grad × Hispanic'),
    ('C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.5]',  'College Grad × Other'),
    ('C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.2]',  'Grad Ed × Black'),
    ('C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.3]',  'Grad Ed × Asian'),
    ('C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.4]',  'Grad Ed × Hispanic'),
    ('C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.5]',  'Grad Ed × Other'),
]

models     = [model1, model2, model3, model4, model5, model6]
col_names  = ['Variable', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']

# Determine control indicators for each model
state_ctrl = ['No', 'Yes',
              'Yes' if m3_has_state else 'No',
              'Yes' if m3_has_state else 'No',
              'Yes' if m5_has_state else 'No',
              'No']
occ_ctrl   = ['No', 'Yes',
              'Yes' if m3_has_occ else 'No',
              'Yes' if m3_has_occ else 'No',
              'Yes' if m5_has_occ else 'No',
              'Yes']
ind_ctrl   = ['No', 'Yes',
              'Yes' if m3_has_ind else 'No',
              'Yes' if m3_has_ind else 'No',
              'Yes' if m5_has_ind else 'No',
              'Yes']

def build_results_table():
    rows = []
    for param, label in PARAM_ROWS:
        # Only include rows where at least one model has this parameter
        cells = [get_cell(m, param) for m in models]
        if all(c[0] == '' for c in cells):
            continue
        coef_row = [label]  + [c[0] for c in cells]
        se_row   = ['']     + [c[1] for c in cells]
        rows.append(coef_row)
        rows.append(se_row)

    # Footer rows
    rows.append([''] * 7)
    rows.append(['Controls for State?']      + state_ctrl)
    rows.append(['Controls for Occupation?'] + occ_ctrl)
    rows.append(['Controls for Industry?']   + ind_ctrl)
    rows.append(['N'] + [f"{int(m.nobs):,}" for m in models])
    rows.append(['Adj. R²'] + [f"{m.rsquared_adj:.4f}" for m in models])

    return pd.DataFrame(rows, columns=col_names)


results_table = build_results_table()
results_table.to_excel("Results.xlsx", index=False)
print("\nResults table saved to Results.xlsx")

# Print key Model 4 F-test summary
print("\n=== Model 4: Interaction F-Test Results ===")
print(f"Q1 - Does the gender gap depend on education?")
print(f"     H0: all Male x Education coefficients = 0")
print(f"     F = {fstat_q1:.4f}, p = {pval_q1:.4f}  => "
      f"{'Reject H0' if pval_q1 < 0.05 else 'Fail to reject H0'} at 5%")
print(f"Q2 - Does the gender gap depend on race?")
print(f"     H0: all Male x Race coefficients = 0")
print(f"     F = {fstat_q2:.4f}, p = {pval_q2:.4f}  => "
      f"{'Reject H0' if pval_q2 < 0.05 else 'Fail to reject H0'} at 5%")
print(f"Q3 - Does the return to education depend on race?")
print(f"     H0: all Education x Race coefficients = 0")
print(f"     F = {fstat_q3:.4f}, p = {pval_q3:.4f}  => "
      f"{'Reject H0' if pval_q3 < 0.05 else 'Fail to reject H0'} at 5%")
