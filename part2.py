"""
@file       part2.py
@author     Finley Clapper
@course     Applied Econometrics I (ECON: 325)
@institution The University of Akron
@date       Spring 2026
@description Part 2 - Identify the Determinants of Income
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

#initialize db (fresh copy so part 1 label mapping doesnt affect regressions)
db = pd.read_csv(os.path.join(os.getcwd(), "DB1.csv"))

#helper to run a joint f-test on a list of parameters
def f_test(model, params):
    matched = [p for p in params if p in model.params.index]
    if not matched:
        return None, 1.0
    all_params = list(model.params.index)
    R = np.zeros((len(matched), len(all_params)))
    for j, p in enumerate(matched):
        R[j, all_params.index(p)] = 1
    result = model.f_test(R)
    return float(result.fvalue), float(result.pvalue)

#helper to get the main effect dummy names for a categorical variable
def cat_dummies(model, var):
    return [p for p in model.params.index if f'C({var}' in p and ':' not in p]

#generate and return model 1 (baseline - no state, occ, or industry)
def gen_model1(db: pd.DataFrame):
    model = smf.ols(
        "income ~ male + C(raceethnic, Treatment(1)) + C(ed, Treatment(1)) + "
        "C(marst, Treatment(1)) + nchild + wkswork + age",
        data=db
    ).fit(cov_type='HC1')
    return model

#generate model 1
model1 = gen_model1(db)

#generate and return model 2 (add state, occupation, and industry controls)
def gen_model2(db: pd.DataFrame):
    model = smf.ols(
        "income ~ male + C(raceethnic, Treatment(1)) + C(ed, Treatment(1)) + "
        "C(marst, Treatment(1)) + nchild + wkswork + age + "
        "C(statefips, Treatment(1)) + C(occ, Treatment(1)) + C(ind, Treatment(1))",
        data=db
    ).fit(cov_type='HC1')
    return model

#generate model 2
model2 = gen_model2(db)

#generate and return model 3 (drop variables insignificant at 5% from model 2)
def gen_model3(db: pd.DataFrame, model2):

    #test continuous/discrete variables with individual t-tests
    cont_vars = ['male', 'nchild', 'wkswork', 'age']
    sig_cont = [v for v in cont_vars if model2.pvalues.get(v, 1.0) < 0.05]

    #test categorical variable groups with joint f-tests
    cat_vars = ['raceethnic', 'ed', 'marst', 'statefips', 'occ', 'ind']
    sig_cat = [v for v in cat_vars if f_test(model2, cat_dummies(model2, v))[1] < 0.05]

    #build formula from significant variables only
    terms = sig_cont + [f"C({v}, Treatment(1))" for v in sig_cat]
    model = smf.ols("income ~ " + " + ".join(terms), data=db).fit(cov_type='HC1')
    return model, sig_cont, sig_cat

#generate model 3
model3, sig_cont_m3, sig_cat_m3 = gen_model3(db, model2)

#generate and return model 4 (add three interaction sets to model 3)
def gen_model4(db: pd.DataFrame, model3, sig_cont, sig_cat):

    #rebuild model 3 terms
    terms = sig_cont + [f"C({v}, Treatment(1))" for v in sig_cat]

    #ensure main effects needed for interactions are present
    for needed in ['male', 'C(raceethnic, Treatment(1))', 'C(ed, Treatment(1))']:
        if needed not in terms:
            terms.append(needed)

    #add the three interaction sets
    terms.append('male:C(ed, Treatment(1))')           # q1: gender gap by education
    terms.append('male:C(raceethnic, Treatment(1))')   # q2: gender gap by race
    terms.append('C(ed, Treatment(1)):C(raceethnic, Treatment(1))')  # q3: return to ed by race

    model = smf.ols("income ~ " + " + ".join(terms), data=db).fit(cov_type='HC1')

    #run joint f-tests for each interaction set
    m_ed_params    = [p for p in model.params.index if 'male' in p and 'C(ed' in p]
    m_race_params  = [p for p in model.params.index if 'male' in p and 'C(raceethnic' in p]
    ed_race_params = [p for p in model.params.index if 'C(ed' in p and 'C(raceethnic' in p]

    fstat_q1, pval_q1 = f_test(model, m_ed_params)
    fstat_q2, pval_q2 = f_test(model, m_race_params)
    fstat_q3, pval_q3 = f_test(model, ed_race_params)

    return model, terms, fstat_q1, pval_q1, fstat_q2, pval_q2, fstat_q3, pval_q3

#generate model 4
model4, m4_terms, fstat_q1, pval_q1, fstat_q2, pval_q2, fstat_q3, pval_q3 = gen_model4(db, model3, sig_cont_m3, sig_cat_m3)

#print f-test results for model 4 interactions
print("\n=== Model 4: Interaction F-Test Results ===")
print(f"Q1 - Does the gender gap depend on education?")
print(f"     H0: all Male x Education coefficients = 0")
print(f"     F = {fstat_q1:.4f}, p = {pval_q1:.4f}  =>  {'Reject H0' if pval_q1 < 0.05 else 'Fail to reject H0'} at 5%")
print(f"Q2 - Does the gender gap depend on race?")
print(f"     H0: all Male x Race coefficients = 0")
print(f"     F = {fstat_q2:.4f}, p = {pval_q2:.4f}  =>  {'Reject H0' if pval_q2 < 0.05 else 'Fail to reject H0'} at 5%")
print(f"Q3 - Does the return to education depend on race?")
print(f"     H0: all Education x Race coefficients = 0")
print(f"     F = {fstat_q3:.4f}, p = {pval_q3:.4f}  =>  {'Reject H0' if pval_q3 < 0.05 else 'Fail to reject H0'} at 5%")

#generate and return model 5 (drop insignificant variables from model 4 at 5%)
def gen_model5(db: pd.DataFrame, model4, m4_terms, pval_q1, pval_q2, pval_q3):

    #get the main effect terms from model 4 (exclude interaction terms)
    main_terms = [t for t in m4_terms if ':' not in t]
    terms = []

    #re-test each main effect using model 4 estimates
    for term in main_terms:
        if term.startswith('C('):
            var = term.split('(')[1].split(',')[0]
            _, p = f_test(model4, cat_dummies(model4, var))
            if p < 0.05:
                terms.append(term)
        else:
            if model4.pvalues.get(term, 1.0) < 0.05:
                terms.append(term)

    #add back interaction sets that are jointly significant
    if pval_q1 < 0.05:
        if 'male' not in terms: terms.append('male')
        if 'C(ed, Treatment(1))' not in terms: terms.append('C(ed, Treatment(1))')
        terms.append('male:C(ed, Treatment(1))')

    if pval_q2 < 0.05:
        if 'male' not in terms: terms.append('male')
        if 'C(raceethnic, Treatment(1))' not in terms: terms.append('C(raceethnic, Treatment(1))')
        terms.append('male:C(raceethnic, Treatment(1))')

    if pval_q3 < 0.05:
        if 'C(ed, Treatment(1))' not in terms: terms.append('C(ed, Treatment(1))')
        if 'C(raceethnic, Treatment(1))' not in terms: terms.append('C(raceethnic, Treatment(1))')
        terms.append('C(ed, Treatment(1)):C(raceethnic, Treatment(1))')

    model = smf.ols("income ~ " + " + ".join(terms), data=db).fit(cov_type='HC1')
    return model, terms

#generate model 5
model5, m5_terms = gen_model5(db, model4, m4_terms, pval_q1, pval_q2, pval_q3)

#generate and return cleaned ipums data for pennsylvania 2023
def gen_ipums_data(path: str) -> pd.DataFrame:

    #load raw ipums file
    raw = pd.read_csv(path)

    #filter to pennsylvania 2023
    pa = raw[(raw['STATEFIP'] == 42) & (raw['YEAR'] == 2023)].copy()

    #recode sex to male indicator
    pa['male'] = (pa['SEX'] == 1).astype(int)

    #recode race and hispanic origin to 5-category raceethnic
    def map_race(race, hispan):
        if hispan > 0:         return 4  # hispanic
        if race == 1:          return 1  # white
        if race == 2:          return 2  # black
        if race in [4, 5, 6]:  return 3  # asian / pacific islander
        return 5                         # other

    pa['raceethnic'] = pa.apply(lambda r: map_race(r['RACE'], r['HISPAN']), axis=1)

    #recode ipums educ (0-11) to 5-category ed
    def map_ed(educ):
        if educ <= 5:  return 1  # no hs diploma
        if educ == 6:  return 2  # hs graduate
        if educ <= 9:  return 3  # some college
        if educ == 10: return 4  # college graduate
        return 5                 # graduate education

    pa['ed'] = pa['EDUC'].apply(map_ed)

    #map detailed ipums occ codes to 16 consolidated categories
    def map_occ(occ):
        if occ == 0:              return 0   # n/a
        if 10   <= occ <= 430:    return 1   # management
        if 500  <= occ <= 740:    return 2   # business & finance
        if 800  <= occ <= 1240:   return 3   # computer, math & engineering
        if 1300 <= occ <= 1560:   return 4   # life, physical & social science
        if 1600 <= occ <= 1960:   return 5   # community, social service & legal
        if 2000 <= occ <= 2060:   return 6   # education
        if 2100 <= occ <= 2180:   return 7   # arts, entertainment & media
        if 2200 <= occ <= 2550:   return 8   # healthcare practitioners
        if 2600 <= occ <= 2970:   return 9   # healthcare support
        if 3000 <= occ <= 3550:   return 10  # protective service
        if 3600 <= occ <= 3960:   return 11  # food prep, cleaning & personal care
        if 4000 <= occ <= 4660:   return 12  # sales, office & admin support
        if 4700 <= occ <= 5940:   return 13  # farming, construction & mining
        if 6000 <= occ <= 7620:   return 14  # production, transport & repair
        if 9800 <= occ <= 9830:   return 15  # military
        if occ  == 9920:          return 16  # unemployed / last job
        return 0

    #map detailed ipums ind codes to 15 consolidated categories
    def map_ind(ind):
        if ind == 0:               return 0   # n/a
        if 170  <= ind <= 290:     return 1   # agriculture, forestry & fishing
        if 370  <= ind <= 490:     return 2   # mining
        if ind  == 570:            return 3   # construction
        if 1070 <= ind <= 3990:    return 4   # manufacturing
        if 4070 <= ind <= 4590:    return 5   # wholesale trade
        if 4670 <= ind <= 5790:    return 6   # retail trade
        if 6070 <= ind <= 6390:    return 7   # transportation & warehousing
        if 6470 <= ind <= 6780:    return 8   # information
        if 6870 <= ind <= 7190:    return 9   # finance & insurance
        if 7270 <= ind <= 7790:    return 10  # professional & real estate services
        if 7860 <= ind <= 8470:    return 11  # educational services
        if 8560 <= ind <= 8690:    return 12  # healthcare & social assistance
        if 8770 <= ind <= 9290:    return 13  # arts, entertainment & food services
        if 9370 <= ind <= 9590:    return 14  # other services
        if 9670 <= ind <= 9870:    return 15  # public administration
        return 0

    #apply occ and ind mappings
    pa['occ'] = pa['OCC'].apply(map_occ)
    pa['ind'] = pa['IND'].apply(map_ind)

    #rename remaining variables to match db1
    pa['marst']   = pa['MARST']
    pa['nchild']  = pa['NCHILD']
    pa['wkswork'] = pa['WKSWORK1']
    pa['age']     = pa['AGE']
    pa['income']  = pa['INCWAGE']

    #keep only workers with positive non-missing wage income and a valid occupation
    pa_clean = pa[
        (pa['income'] > 0) & (pa['income'] < 999998) &
        (pa['wkswork'] > 0) &
        (pa['occ'] > 0) & (pa['ind'] > 0)
    ].copy()

    return pa_clean

#generate cleaned ipums data for pennsylvania 2023
ipums_path = os.path.join(os.getcwd(), "usa_00002.csv")
pa_clean = gen_ipums_data(ipums_path)
print(f"\nIPUMS PA 2023 sample: {len(pa_clean):,} observations")
print(f"Average income (INCWAGE): ${pa_clean['income'].mean():,.2f}")

#generate and return model 6 (ipums pennsylvania 2023 with all controls and interactions)
def gen_model6(pa_clean: pd.DataFrame):
    model = smf.ols(
        "income ~ male + C(raceethnic, Treatment(1)) + C(ed, Treatment(1)) + "
        "C(marst, Treatment(1)) + nchild + wkswork + age + "
        "C(occ, Treatment(1)) + C(ind, Treatment(1)) + "
        "male:C(raceethnic, Treatment(1)) + "
        "male:C(ed, Treatment(1)) + "
        "C(ed, Treatment(1)):C(raceethnic, Treatment(1))",
        data=pa_clean
    ).fit(cov_type='HC1')
    return model

#generate model 6
model6 = gen_model6(pa_clean)

#generate and return combined regression results table (all 6 models)
def gen_results_table(model1, model2, model3, model4, model5, model6,
                      sig_cat_m3, m5_terms) -> pd.DataFrame:

    #ordered parameter names and their display labels (state/occ/ind excluded)
    param_labels = {
        'Intercept':                                                    'Intercept',
        'male':                                                         'Male',
        'C(raceethnic, Treatment(1))[T.2]':                            'Black',
        'C(raceethnic, Treatment(1))[T.3]':                            'Asian',
        'C(raceethnic, Treatment(1))[T.4]':                            'Hispanic',
        'C(raceethnic, Treatment(1))[T.5]':                            'Other Race',
        'C(ed, Treatment(1))[T.2]':                                    'HS Graduate',
        'C(ed, Treatment(1))[T.3]':                                    'Some College',
        'C(ed, Treatment(1))[T.4]':                                    'College Graduate',
        'C(ed, Treatment(1))[T.5]':                                    'Graduate Education',
        'C(marst, Treatment(1))[T.2]':                                 'Married, Spouse Absent',
        'C(marst, Treatment(1))[T.3]':                                 'Separated',
        'C(marst, Treatment(1))[T.4]':                                 'Divorced',
        'C(marst, Treatment(1))[T.5]':                                 'Widowed',
        'C(marst, Treatment(1))[T.6]':                                 'Never Married',
        'nchild':                                                       'Nchild',
        'wkswork':                                                      'Weeks Worked',
        'age':                                                          'Age',
        'male:C(ed, Treatment(1))[T.2]':                               'Male x HS Graduate',
        'male:C(ed, Treatment(1))[T.3]':                               'Male x Some College',
        'male:C(ed, Treatment(1))[T.4]':                               'Male x College Graduate',
        'male:C(ed, Treatment(1))[T.5]':                               'Male x Graduate Education',
        'male:C(raceethnic, Treatment(1))[T.2]':                       'Male x Black',
        'male:C(raceethnic, Treatment(1))[T.3]':                       'Male x Asian',
        'male:C(raceethnic, Treatment(1))[T.4]':                       'Male x Hispanic',
        'male:C(raceethnic, Treatment(1))[T.5]':                       'Male x Other Race',
        'C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.2]':  'HS Grad x Black',
        'C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.3]':  'HS Grad x Asian',
        'C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.4]':  'HS Grad x Hispanic',
        'C(ed, Treatment(1))[T.2]:C(raceethnic, Treatment(1))[T.5]':  'HS Grad x Other',
        'C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.2]':  'Some College x Black',
        'C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.3]':  'Some College x Asian',
        'C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.4]':  'Some College x Hispanic',
        'C(ed, Treatment(1))[T.3]:C(raceethnic, Treatment(1))[T.5]':  'Some College x Other',
        'C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.2]':  'College Grad x Black',
        'C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.3]':  'College Grad x Asian',
        'C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.4]':  'College Grad x Hispanic',
        'C(ed, Treatment(1))[T.4]:C(raceethnic, Treatment(1))[T.5]':  'College Grad x Other',
        'C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.2]':  'Grad Ed x Black',
        'C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.3]':  'Grad Ed x Asian',
        'C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.4]':  'Grad Ed x Hispanic',
        'C(ed, Treatment(1))[T.5]:C(raceethnic, Treatment(1))[T.5]':  'Grad Ed x Other',
    }

    models = [model1, model2, model3, model4, model5, model6]
    rows = []

    #build coefficient and SE rows for each parameter
    for param, label in param_labels.items():

        #skip rows where no model has this parameter
        if not any(param in m.params.index for m in models):
            continue

        coef_row = [label]
        se_row   = ['']

        for m in models:
            if param in m.params.index:
                stars = ''
                p = m.pvalues[param]
                if p <= 0.01:  stars = '***'
                elif p <= 0.05: stars = '**'
                elif p <= 0.10: stars = '*'
                coef_row.append(f"{m.params[param]:,.2f}{stars}")
                se_row.append(f"({m.bse[param]:,.2f})")
            else:
                coef_row.append('')
                se_row.append('')

        rows.append(coef_row)
        rows.append(se_row)

    #add control indicator rows and summary stats
    rows.append([''] * 7)
    rows.append(['Controls for State?',      'No', 'Yes',
                 'Yes' if 'statefips' in sig_cat_m3 else 'No',
                 'Yes' if 'statefips' in sig_cat_m3 else 'No',
                 'Yes' if 'C(statefips, Treatment(1))' in m5_terms else 'No',
                 'No'])
    rows.append(['Controls for Occupation?', 'No', 'Yes',
                 'Yes' if 'occ' in sig_cat_m3 else 'No',
                 'Yes' if 'occ' in sig_cat_m3 else 'No',
                 'Yes' if 'C(occ, Treatment(1))' in m5_terms else 'No',
                 'Yes'])
    rows.append(['Controls for Industry?',   'No', 'Yes',
                 'Yes' if 'ind' in sig_cat_m3 else 'No',
                 'Yes' if 'ind' in sig_cat_m3 else 'No',
                 'Yes' if 'C(ind, Treatment(1))' in m5_terms else 'No',
                 'Yes'])
    rows.append(['N'] + [f"{int(m.nobs):,}" for m in models])
    rows.append(['Adj. R2'] + [f"{m.rsquared_adj:.4f}" for m in models])

    cols = ['Variable', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']
    return pd.DataFrame(rows, columns=cols)

#generate results table and export to excel
results_table = gen_results_table(model1, model2, model3, model4, model5, model6,
                                  sig_cat_m3, m5_terms)
results_table.to_excel("Results.xlsx", index=False)
