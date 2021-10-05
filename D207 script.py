# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 14:03:29 2021

@author: dtwol
"""

import pandas as pd
import scipy
from scipy.stats import kurtosis, skew
from scipy import stats
import matplotlib 
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np
import scipy.stats as stats

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


df = pd.read_csv('C:/Users/dtwol/OneDrive - Western Governors University/D207/medical_clean.csv')
#df.describe()
df.info()


# Function calling basic data strucure of columns
def unistats(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis, skew
            
    output_df = pd.DataFrame(columns=['Unique', 'Type', 'Min', 'Max', 'Std', 'Skew', 'Kurt'])
    
    for col in df.columns:
        # these are the outputs that apply to every variable regardless of data type
        unique = df[col].nunique()
        dtype = str(df[col].dtype)
              
        if pd.api.types.is_numeric_dtype(dtype):
        # perform additional calculations for numeric variables
            min = round(df[col].min(), 2)
            max = round(df[col].max(), 2)
            std = round(df[col].std(), 2) 
            skew = round(df[col].skew(), 2)
            kurt = round(df[col].kurt(), 2)
        else:
            # perform any applicable calculations for categorical variables
            min = '-'
            max = '-'
            std = '-'
            skew = '-'
            kurt = '-'
    
        output_df.loc[col] = (unique, dtype, min, max, std, skew, kurt)
            
    return output_df

# Calling function
unistats(df)




####Distribution plots for univariate categorical variables####
sns.set_style("white")
sns.set(palette=sns.color_palette("RdBu_r", df.Area.nunique())) 
sns.catplot(x="Gender", kind="count", data=df);
sns.catplot(x="Area", kind="count", data=df)
sns.catplot(x="TimeZone", kind="count", data=df)
plt.xticks(rotation=90)
plt.tight_layout()
sns.catplot(x="Marital", kind="count", data=df)
plt.xticks(rotation=45)
sns.catplot(x="Services", kind="count", data=df)


plt.clf() 


# 1. Define the automation function
# def bivariate_stats():
      
# 2. Import python packages

# 3. Create variables needed for processing

# 4. Define the iteration (i.e. for loop)

# 5. Perform processing needed for every iteration

# 6. Define the decision criterion

# 7. Perform processing required in each branch of the decision tree

# 8. Perform any final processing required to synthesize all branches (if any)

df.info()

num_list = []
cat_list = []

for column in df:
    if is_numeric_dtype(df[column]):
        num_list.append(column)
    elif is_string_dtype(df[column]):
        cat_list.append(column)
        
print(cat_list)
print(num_list)


for column in df:
    plt.figure(column)
    plt.title(column)
    if is_numeric_dtype(df[column]):
        df[column].plot(kind='hist')
    elif is_string_dtype(df[column]):
        df[column].value_counts()[:20].plot(kind='bar')
        

for column in df:
    if is_numeric_dtype(df[column]):
        for i in range(0, len(num_list)):
            num1 = num_list[i]
            for j in range(0, len(num_list)):
                num2 = num_list[j]
                plt.figure (figsize = (15,15))
                sns.regplot(x=num1, y=num2)
                sns.despine(top=True, right=True)
                m, b, r, p, err = stats.linregress(x=num1, y=num2)
                # Add the formula, r squared, and p-value to the figure
                textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
                textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
                textstr += 'p  = ' + str(round(p, 2))
                plt.text(0.15, 0.70, textstr, fontsize=12, transform=plt.gcf().transFigure)
                plt.show()











def bivariate_stats(df, label='label_name'):

    df_output = pd.DataFrame(columns=['test', 'stat', 'p-value'])

    for col in df:
        print(col)
















####Distribution plots on univariate numeric variables####

####
plt.hist(df.Population, bins=int(round(df.Population.count()**(1/3), 0)))
text  = 'Std dev: ' + str(round(df.Population.std(), 2)) + '\n'
text += 'Mean: ' + str(round(df.Population.mean(), 2)) + '\n'
text += 'Skew: ' + str(round(df.Population.skew(), 2)) + '\n'
text += 'Kurt: ' + str(round(df.Population.kurt(), 2))
plt.text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.axvline(df.Population.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('# of Population')
plt.xlabel('Population')
plt.ylabel('Count')
plt.show()


####
plt.hist(df.Income, bins=int(round(df.Income.count()**(1/3), 0)))
text  = 'Std dev: ' + str(round(df.Income.std(), 2)) + '\n'
text += 'Mean: ' + str(round(df.Income.mean(), 2)) + '\n'
text += 'Skew: ' + str(round(df.Income.skew(), 2)) + '\n'
text += 'Kurt: ' + str(round(df.Income.kurt(), 2))
plt.text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.axvline(df.Income.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('# of Income')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()


####
plt.hist(df.TotalCharge, bins=int(round(df.TotalCharge.count()**(1/3), 0)))
text  = 'Std dev: ' + str(round(df.TotalCharge.std(), 2)) + '\n'
text += 'Mean: ' + str(round(df.TotalCharge.mean(), 2)) + '\n'
text += 'Skew: ' + str(round(df.TotalCharge.skew(), 2)) + '\n'
text += 'Kurt: ' + str(round(df.TotalCharge.kurt(), 2))
plt.text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.axvline(df.TotalCharge.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('# of TotalCharge')
plt.xlabel('TotalCharge')
plt.ylabel('Count')
plt.show()


####
plt.hist(df.Additional_charges, bins=int(round(df.Additional_charges.count()**(1/3), 0)))
text  = 'Std dev: ' + str(round(df.Additional_charges.std(), 2)) + '\n'
text += 'Mean: ' + str(round(df.Additional_charges.mean(), 2)) + '\n'
text += 'Skew: ' + str(round(df.Additional_charges.skew(), 2)) + '\n'
text += 'Kurt: ' + str(round(df.Additional_charges.kurt(), 2))
plt.text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.axvline(df.Additional_charges.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('# of Additional_charges')
plt.xlabel('Additional_charges')
plt.ylabel('Count')
plt.show()

####
plt.hist(df.Age, bins=int(round(df.Age.count()**(1/3), 0)))
text  = 'Std dev: ' + str(round(df.Age.std(), 2)) + '\n'
text += 'Mean: ' + str(round(df.Age.mean(), 2)) + '\n'
text += 'Skew: ' + str(round(df.Age.skew(), 2)) + '\n'
text += 'Kurt: ' + str(round(df.Age.kurt(), 2))
plt.text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.axvline(df.Age.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('# of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

####
plt.hist(df.Children, bins=int(round(df.Children.count()**(1/3), 0)))
text  = 'Std dev: ' + str(round(df.Children.std(), 2)) + '\n'
text += 'Mean: ' + str(round(df.Children.mean(), 2)) + '\n'
text += 'Skew: ' + str(round(df.Children.skew(), 2)) + '\n'
text += 'Kurt: ' + str(round(df.Children.kurt(), 2))
plt.text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.axvline(df.Children.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('# of Children')
plt.xlabel('Children')
plt.ylabel('Count')
plt.show()




plt.clf() 

####Distribuation plots on bivariate numeric variables####
####
ttlcharge_initdays_plot = sns.regplot(x=df.TotalCharge, y=df.Initial_days)
sns.despine(top=True, right=True)
# Calculate the regression line
m, b, r, p, err = stats.linregress(df.TotalCharge, df.Initial_days)
# Add the formula, r squared, and p-value to the figure
textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
textstr += 'p  = ' + str(round(p, 2))
plt.text(0.15, 0.70, textstr, fontsize=12, transform=plt.gcf().transFigure)
plt.show(ttlcharge_initdays_plot)

####
adtlcharge_age_plot = sns.regplot(x=df.Additional_charges, y=df.Age)
sns.despine(top=True, right=True)
# Calculate the regression line
m, b, r, p, err = stats.linregress(df.Additional_charges, df.Age)
# Add the formula, r squared, and p-value to the figure
textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
textstr += 'p  = ' + str(round(p, 2))
plt.text(0.15, 0.70, textstr, fontsize=12, transform=plt.gcf().transFigure)
plt.show(adtlcharge_age_plot)


####
lat_zip_plot = sns.regplot(x= df.Lat, y= df.Zip)
sns.despine(top=True, right=True)
# Calculate the regression line
m, b, r, p, err = stats.linregress(df.Lat, df.Zip)
# Add the formula, r squared, and p-value to the figure
textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
textstr += 'p  = ' + str(round(p, 2))
plt.text(0.15, 0.70, textstr, fontsize=12, transform=plt.gcf().transFigure)
plt.show(lat_zip_plot)

####
lng_zip_plot = sns.regplot(x= df.Lng, y= df.Zip)
sns.despine(top=True, right=True)
# Calculate the regression line
m, b, r, p, err = stats.linregress(df.Lng, df.Zip)
# Add the formula, r squared, and p-value to the figure
textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
textstr += 'p  = ' + str(round(p, 2))
plt.text(0.15, 0.70, textstr, fontsize=12, transform=plt.gcf().transFigure)
plt.show(lng_zip_plot)

####
lng_lat_plot = sns.regplot(x= df.Lng, y= df.Lat)
sns.despine(top=True, right=True)
# Calculate the regression line
m, b, r, p, err = stats.linregress(df.Lng, df.Lat)
# Add the formula, r squared, and p-value to the figure
textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
textstr += 'p  = ' + str(round(p, 2))
plt.text(0.15, 0.70, textstr, fontsize=12, transform=plt.gcf().transFigure)
plt.show(lng_lat_plot)

df.info()

####Distribuation plots on bivariate categorical variables####
Area_Mar_viz = sns.catplot(x="Area", hue="Marital",
                  kind="count", palette="pastel", data=df,
                  order=['Suburban', 'Urban Area', 'Rural']);
Area_Mar_viz.set_xticklabels(rotation=25)

####
df.Gender.unique()
Area_Gen_viz = sns.catplot(x="Area", hue="Gender",
                  kind="count", palette="pastel", data=df,
                  order=['Suburban', 'Urban Area', 'Rural']);
Area_Gen_viz.set_xticklabels(rotation=25)

####
df.TimeZone.unique()

TZ_Gen_viz = sns.catplot(x="TimeZone", hue="Gender",
    kind="count", palette="pastel", data=df,
    order=['America/Chicago', 'America/New_York', 'America/Los_Angeles', 'America/Indiana/Indianapolis', 
           'America/Detroit', 'America/Denver', 'America/Anchorage', 'America/Phoenix', 
           'America/Boise', 'America/Puerto_Rico']);
##Intentionally omitted values with counts to small to be relevant 
            #'Pacific/Honolulu', 'America/Menominee', 'America/Yakutat', 'America/Nome', 
            # 'America/Kentucky/Louisville', 'America/Indiana/Vincennes', 'America/Toronto', 'America/Indiana/Marengo',
            # 'America/Indiana/Winamac', 'America/Indiana/Tell_City', 'America/Sitka', 'America/Indiana/Knox',
            # 'America/North_Dakota/New_Salem', 'America/Indiana/Vevay', 'America/Adak', 'America/North_Dakota/Beulah']);
TZ_Gen_viz.set_xticklabels(rotation=90)


####
TZ_Mar_viz = sns.catplot(x="TimeZone", hue="Marital",
    kind="count", palette="pastel", data=df,
    order=['America/Chicago', 'America/New_York', 'America/Los_Angeles', 'America/Indiana/Indianapolis', 
           'America/Detroit', 'America/Denver', 'America/Anchorage', 'America/Phoenix', 
           'America/Boise', 'America/Puerto_Rico', 
           'Pacific/Honolulu']);
TZ_Mar_viz.set_xticklabels(rotation=90)

####
TZ_Mar_viz = sns.catplot(x="TimeZone", hue="ReAdmis",
    kind="count", palette="pastel", data=df,
    order=['America/Chicago', 'America/New_York', 'America/Los_Angeles', 'America/Indiana/Indianapolis', 
           'America/Detroit', 'America/Denver', 'America/Anchorage', 'America/Phoenix', 
           'America/Boise', 'America/Puerto_Rico', 
           'Pacific/Honolulu']);
TZ_Mar_viz.set_xticklabels(rotation=90)

####
TZ_Mar_viz = sns.catplot(x="ReAdmis", hue="Marital",
    kind="count", palette="pastel", data=df,
    order=['No', 'Yes']);
TZ_Mar_viz.set_xticklabels(rotation=0)

####
TZ_Mar_viz = sns.catplot(x="ReAdmis", hue="Area",
    kind="count", palette="pastel", data=df,
    order=['No', 'Yes']);
TZ_Mar_viz.set_xticklabels(rotation=0)

####
TZ_Mar_viz = sns.catplot(x="ReAdmis", hue="Gender",
    kind="count", palette="pastel", data=df,
    order=['No', 'Yes']);
TZ_Mar_viz.set_xticklabels(rotation=0)


####
df['Job'].value_counts().head(45)

####
Job_ReAd_viz = sns.catplot(x="Job", hue="ReAdmis",
    kind="count", palette="pastel", data=df,
    order=['Outdoor activities/education manager', 'Exhibition designer', 'Theatre director', 'Scientist, audiological', 
           'Toxicologist', 'Research scientist (life sciences)', 'Orthoptist', 'Technical sales engineer', 'Astronomer', 
           'Estate agent', 'Marketing executive', 'Production assistant, radio', 'Neurosurgeon', 'Lobbyist']); 
Job_ReAd_viz.set_xticklabels(rotation=90)                   
           
Job_ReAd_viz2 = sns.catplot(x="Job", hue="ReAdmis",
    kind="count", palette="pastel", data=df,
    order=['Jewellery designer', 'Broadcast presenter', 'Leisure centre manager', 'Development worker, international aid',
           'Pharmacist, community', 'Financial trader', 'Physicist, medical', 'Adult nurse', 
           'Chartered legal executive (England and Wales)', 'Surveyor, minerals', 'Energy engineer', 'Firefighter']);
Job_ReAd_viz2.set_xticklabels(rotation=90)

Job_ReAd_viz3 = sns.catplot(x="Job", hue="ReAdmis",
    kind="count", palette="pastel", data=df,
    order=['Research scientist (physical sciences)', 'Chiropodist', 'Arboriculturist', 'Radiographer, diagnostic',
           'Psychologist, forensic', 'Engineer, water', 'Designer, exhibition/display', 'Teacher, secondary school', 
           'Further education lecturer', 'Diplomatic Services operational officer', 'Newspaper journalist', 'Aid worker']);
Job_ReAd_viz3.set_xticklabels(rotation=90)


####heatmap, then contingency Crosstab and then comparison Chi-Square crostab on categorical variables####

####start heatmap crosstab + contingency table + chi-2 comparison table
sns.heatmap(pd.crosstab(df.Gender, df.ReAdmis), annot=True, fmt='d');

Gen_ReAd_crosstab = pd.crosstab(df['Gender'], df['ReAdmis'])
plt.title('CrossTab of Gender and ReAdmis')
sns.heatmap(Gen_ReAd_crosstab, annot=True, fmt='d', cmap='coolwarm');

X, p, dof, contingency_table = chi2_contingency(Gen_ReAd_crosstab)
textstr  = 'X2: ' + str(round(X, 4))+ '\n'
textstr += 'p = ' + str(round(p, 4)) + '\n'
textstr += 'dof  = ' + str(dof)
plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

gen_read_ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=Gen_ReAd_crosstab.columns, index=Gen_ReAd_crosstab.index)
plt.title('Contingency Table of Gender and Readmis')
sns.heatmap(gen_read_ct_df, annot=True, fmt='d', cmap='coolwarm');
####end


####start heatmap crosstab + contingency table + chi-2 comparison table
sns.heatmap(pd.crosstab(df.Marital, df.ReAdmis), annot=True, fmt='d');

Mar_ReAd_crosstab = pd.crosstab(df['Marital'], df['ReAdmis'])
plt.title('CrossTab of Marital and ReAdmis')
sns.heatmap(Mar_ReAd_crosstab, annot=True, fmt='d', cmap='coolwarm');

X, p, dof, contingency_table = chi2_contingency(Mar_ReAd_crosstab)
textstr  = 'X2: ' + str(round(X, 4))+ '\n'
textstr += 'p = ' + str(round(p, 4)) + '\n'
textstr += 'dof  = ' + str(dof)
plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

mar_read_ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), 
                              columns=Mar_ReAd_crosstab.columns, index=Mar_ReAd_crosstab.index)
plt.title('Contingency Table of Marriage and Readmis')
sns.heatmap(mar_read_ct_df, annot=True, fmt='d', cmap='coolwarm');
####end


####start heatmap crosstab + contingency table + chi-2 comparison table
sns.heatmap(pd.crosstab(df.Area, df.ReAdmis), annot=True, fmt='d');

Area_ReAd_crosstab = pd.crosstab(df['Area'], df['ReAdmis'])
plt.title('CrossTab of Area and ReAdmis')
sns.heatmap(Area_ReAd_crosstab, annot=True, fmt='d', cmap='coolwarm');

X, p, dof, contingency_table = chi2_contingency(Area_ReAd_crosstab)
textstr  = 'X2: ' + str(round(X, 4))+ '\n'
textstr += 'p = ' + str(round(p, 4)) + '\n'
textstr += 'dof  = ' + str(dof)
plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

area_read_ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), 
                              columns=Area_ReAd_crosstab.columns, index=Area_ReAd_crosstab.index)
plt.title('Contingency Table of Area and Readmis')
sns.heatmap(area_read_ct_df, annot=True, fmt='d', cmap='coolwarm');
####end


####start heatmap crosstab + contingency table + chi-2 comparison table
sns.heatmap(pd.crosstab(df.TimeZone, df.ReAdmis), annot=True, fmt='d');

TZ_ReAd_crosstab = pd.crosstab(df['TimeZone'], df['ReAdmis'])
plt.title('CrossTab of Timezone and ReAdmis')
sns.heatmap(TZ_ReAd_crosstab, annot=True, fmt='d', cmap='coolwarm');

X, p, dof, contingency_table = chi2_contingency(TZ_ReAd_crosstab)
textstr  = 'X2: ' + str(round(X, 4))+ '\n'
textstr += 'p = ' + str(round(p, 4)) + '\n'
textstr += 'dof  = ' + str(dof)
plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

TZ_read_ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), 
                              columns=TZ_ReAd_crosstab.columns, index=TZ_ReAd_crosstab.index)
plt.title('Contingency Table of timezone and Readmis')
sns.heatmap(TZ_read_ct_df, annot=True, fmt='d', cmap='coolwarm');
####end


####start heatmap crosstab + contingency table + chi-2 comparison table
sns.heatmap(pd.crosstab(df.Initial_admin, df.ReAdmis), annot=True, fmt='d');

InitAd_ReAd_crosstab = pd.crosstab(df['Initial_admin'], df['ReAdmis'])
plt.title('CrossTab of Initial Admin and ReAdmis')
sns.heatmap(InitAd_ReAd_crosstab, annot=True, fmt='d', cmap='coolwarm');

X, p, dof, contingency_table = chi2_contingency(InitAd_ReAd_crosstab)
textstr  = 'X2: ' + str(round(X, 4))+ '\n'
textstr += 'p = ' + str(round(p, 4)) + '\n'
textstr += 'dof  = ' + str(dof)
plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

initad_read_ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), 
                              columns=InitAd_ReAd_crosstab.columns, index=InitAd_ReAd_crosstab.index)
plt.title('Contingency Table of Initial Admin and Readmis')
sns.heatmap(initad_read_ct_df, annot=True, fmt='d', cmap='coolwarm');
####end


####start heatmap crosstab + contingency table + chi-2 comparison table
sns.heatmap(pd.crosstab(df.Services, df.ReAdmis), annot=True, fmt='d');

Serv_ReAd_crosstab = pd.crosstab(df['Services'], df['ReAdmis'])
plt.title('CrossTab of Services and ReAdmis')
sns.heatmap(Serv_ReAd_crosstab, annot=True, fmt='d', cmap='coolwarm');

X, p, dof, contingency_table = chi2_contingency(Serv_ReAd_crosstab)
textstr  = 'X2: ' + str(round(X, 4))+ '\n'
textstr += 'p = ' + str(round(p, 4)) + '\n'
textstr += 'dof  = ' + str(dof)
plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

Serv_read_ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), 
                              columns=Serv_ReAd_crosstab.columns, index=Serv_ReAd_crosstab.index)
plt.title('Contingency Table of services and Readmis')
sns.heatmap(Serv_read_ct_df, annot=True, fmt='d', cmap='coolwarm');
####end



#If you choose Chi-Square, you create a FOR loop and test against each of the categoricals.

categorical_vars = ["State", "Area", "Timezone", "Area", "Job", "Marital", "Gender", "Interaction", "Customer_id",
                    "HighBlood", "Stroke", "Complication_risk", "Overweight", "Arthritis", "Diabetes", 
                    "Hyperlipidemia", "BackPain", "Anxiety", "Allergic_rhinitis", "Reflux_esophagitis", 
                    "Asthma", "Services", "Initial_admin", "Soft_drink", "ReAdmis", "County", "City", "UID"]

chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(y,x))




from scipy.stats import chisquare

df['p'] = chisquare(df[['May', 'June', 'July']], axis=1)[1]

df['same_diff'] = np.where(df['p'] > 0.05, 'same', 'different')