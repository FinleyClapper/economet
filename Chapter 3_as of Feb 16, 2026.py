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
from tabulate import tabulate ##This allows tabulate function to be used directly, as opposed to having to use "tabulate.tabulate".
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Read the data file: See Note 1 in the READ ME section above.
CPSCH3 = pd.read_excel(os.path.join(os.getcwd(), "CPS_Ch3.xlsx"), sheet_name="Data")

# Quick check to ensure data is read/imported properly.
print(CPSCH3.head(7))

# Summary Statistics for Categorical Variables
SumStat_Cat_Rows=[]

for Var in ['Year', 'Sex']:
    Freq=CPSCH3[Var].value_counts().sort_index()
    Percent=(Freq/Freq.sum())*100
    for Cat in Freq.index:
        SumStat_Cat_Rows.append([
            Var,
            str(Cat),
            Freq[Cat],
            Percent[Cat]
        ])

SumStat_Cat_Table = pd.DataFrame(SumStat_Cat_Rows, columns=[
    'Variable',
    'Category',
    'Frequency',
    'Percent'
])
SumStat_Cat_Table['Frequency']=(SumStat_Cat_Table['Frequency'].map(lambda x: f"{x:,}"))

SumStat_Cat_Table['Percent']=(SumStat_Cat_Table['Percent']
                              .map(lambda x: f"{x:.2f}%"))

SumStat_Cat_Table['Variable']=(SumStat_Cat_Table['Variable']
                               .replace({'Sex':'Gender'}))

SumStat_Cat_Table.loc[SumStat_Cat_Table['Variable']=='Gender','Category']=(
    SumStat_Cat_Table.loc[SumStat_Cat_Table['Variable']=='Gender','Category']
    .map({"1": "Male", "2": "Female"}))

print(SumStat_Cat_Table)

SumStat_Cat_Table.to_excel("SumStat_Cat_Table.xlsx", index=False)

# Summary Statistics for Continuous Variables
SummaryAHE=CPSCH3['AHE15'].agg(['count', 'mean', 'std', 'min', 'max'])
print(SummaryAHE)

SummaryAHE_YearSex=CPSCH3.groupby(['Year', 'Sex'])['AHE15'].agg(
    Obs='count',
    Mean='mean',
    Std='std',
    Min='min',
    Max='max',
).reset_index()
print(SummaryAHE_YearSex)

Edited_SummaryAHE_YearSex=(SummaryAHE_YearSex.assign(
    Sex=lambda df: df['Sex'].map({1:'Male', 2:'Female'}),
    Mean=lambda df: df['Mean'].map(lambda x: f"${x:,.2f}"),
    Std=lambda df: df['Std'].map(lambda x: f"${x:,.2f}"),
    Min=lambda df: df['Min'].map(lambda x: f"${x:,.2f}"),
    Max=lambda df: df['Max'].map(lambda x: f"${x:,.2f}"),
    Obs=lambda df: df['Obs'].map(lambda x: f"{x:,.0f}"),
).rename(columns={
    'Sex':'Gender',
    'Obs':"Observations",
    'Mean':"Average",
    'Std':'Standard Deviation',
    'Min':'Minimum',
    'Max':'Maximum',
})
)
print(Edited_SummaryAHE_YearSex)

print(tabulate(Edited_SummaryAHE_YearSex,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        stralign='center',
        numalign='center',
))

Edited_SummaryAHE_YearSex.to_excel("SummaryAHE_YearSex.xlsx",
                                   index=False)

# T-Test Table
TTable_Rows=[]

for year in sorted(CPSCH3['Year'].unique()):
    data_year=CPSCH3[CPSCH3['Year'] == year]
    men=data_year[data_year['Sex']==1]['AHE15']
    women=data_year[data_year['Sex']==2]['AHE15']

    mean_men=men.mean()
    mean_women=women.mean()

    diff=mean_men-mean_women

    # results=ttest_ind(men,women,equal_var=False)
    p_value=ttest_ind(men,women,equal_var=False).pvalue

    if p_value <= 0.01:
        stars = '***'
    elif p_value <= 0.05:
        stars = '**'
    elif p_value <= 0.10:
        stars = '*'
    else:
        stars=''

    mean_men_fmt=f"${mean_men:.2f}"
    mean_women_fmt=f"${mean_women:.2f}"
    diff_fmt=f"${diff:.2f}{stars}"

    TTable_Rows.append([year,mean_men_fmt,mean_women_fmt,diff_fmt ])

TTable=pd.DataFrame(TTable_Rows,columns=['Year',
                                         "Men's Avg. AHE",
                                         "Women's Avg. AHE",
                                         'Difference(Men-Women)'])

print(TTable)
print(tabulate(TTable,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        stralign='center',
        numalign='center',
))
TTable.to_excel("TTable.xlsx", index=False)

# Bar Chart: AHE by Year
avg_by_year=CPSCH3.groupby('Year')['AHE15'].mean()

print(avg_by_year)

#plt.bar(avg_by_year.index,avg_by_year.values)
x_ticks=range(len(avg_by_year))
print(list(x_ticks))

plt.bar(x_ticks,avg_by_year.values, width=0.5)
plt.xlabel("Year", fontsize=13, fontweight='bold')
plt.xticks(x_ticks,avg_by_year.index)
plt.ylabel("AHE in 2015 Dollars", fontsize=13, fontweight='bold')
plt.title('Figure 1. Average Hourly Earnings by Year', fontsize=14, fontweight='bold')

for i, value in enumerate(avg_by_year):
    plt.text(i, value, f'${value:,.2f}', ha='center', va='bottom')

plt.tight_layout()

plt.savefig("AHE_by_Year.jpg", dpi=300)

plt.show()