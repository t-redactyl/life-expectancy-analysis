# Import all necessary libraries
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.linear_model import Lasso

# Import life expectancy and predictor data from World Health Organisation Global
# Health Observatory data repo (http://www.who.int/gho/en/)
# Downloaded on 24th July, 2016

def dataImport(dataurl):
	url = dataurl
	return pd.read_csv(url)

# 1. Life expectancy (from: http://apps.who.int/gho/data/node.main.688?lang=en)
	# Multiple years, used 2015
	# Used both sexes
life = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/WHOSIS_000001,WHOSIS_000015&profile=crosstable&filter=COUNTRY:*&x-sideaxis=COUNTRY;YEAR&x-topaxis=GHO;SEX")

# 2. Alcohol (from: http://apps.who.int/gho/data/node.main.A1026?lang=en)
	# Multiple years, used 2010
	# Used 'all types' under Beverage Types
alcohol = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/SA_0000001400&profile=crosstable&filter=COUNTRY:*;YEAR:2015;YEAR:2014;YEAR:2013;YEAR:2012;YEAR:2011;YEAR:2010;YEAR:2009;YEAR:2008;YEAR:2007;YEAR:2006;YEAR:2005;YEAR:2004;YEAR:2003;YEAR:2002;YEAR:2001;YEAR:2000&x-sideaxis=COUNTRY;DATASOURCE;ALCOHOLTYPE&x-topaxis=GHO;YEAR")

# 3. Overweight (from: http://apps.who.int/gho/data/node.main.A897A?lang=en)
	# Multiple years, 2014 used
	# Used both sexes
oweight = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/NCD_BMI_25A&profile=crosstable&filter=AGEGROUP:*;COUNTRY:*;SEX:*&x-sideaxis=COUNTRY&x-topaxis=GHO;YEAR;AGEGROUP;SEX")

# 4. Cholesterol (from: http://apps.who.int/gho/data/node.main.A884?lang=en)
	# Single year (2008) only
	# Used both sexes
	# Used age standardised estimates
chol = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/CHOL_01,CHOL_02&profile=crosstable&filter=AGEGROUP:*;COUNTRY:*;SEX:*&x-sideaxis=COUNTRY;YEAR;AGEGROUP&x-topaxis=GHO;SEX")

# 5. Blood sugar (from: http://apps.who.int/gho/data/node.main.A869?lang=en)
	# Multiple years, 2014 used
	# Used both sexes
	# Used crude blood sugar estimate
bsugar = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/NCD_GLUC_03,NCD_GLUC_04&profile=crosstable&filter=AGEGROUP:*;COUNTRY:*;SEX:*&x-sideaxis=COUNTRY;YEAR;AGEGROUP&x-topaxis=GHO;SEX")

# 6. Unsafe water, sanitation and hygiene (from: http://apps.who.int/gho/data/node.main.167?lang=en)
	# Multiple years, used 2015
	# Used Total
	# % using improved water sources and % using improved sanitation
sanitation = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/WHS5_122,WHS5_158&profile=crosstable&filter=COUNTRY:*;RESIDENCEAREATYPE:*&x-sideaxis=COUNTRY;YEAR&x-topaxis=GHO;RESIDENCEAREATYPE")

# 7. Maternal deaths (from: http://apps.who.int/gho/data/node.main.MATMORT?lang=en) 
	# Multiple years, used 2015
	# Used maternal mortality ratio (per 100 000 live births)
maternal = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/MDG_0000000026,MORT_MATERNALNUM&profile=crosstable&filter=COUNTRY:*;REGION:*&x-sideaxis=COUNTRY;YEAR&x-topaxis=GHO")

# 8. UV radiation exposure (from: http://apps.who.int/gho/data/node.main.164?lang=en)
	# One year and measure only 
uvrad = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/UV_1&profile=crosstable&filter=COUNTRY:*;REGION:*&x-sideaxis=COUNTRY&x-topaxis=GHO;YEAR")

# 9. Homicide (from: http://apps.who.int/gho/data/node.main.VIOLENCEHOMICIDE?lang=en)
	# Used 2012 (one year only)
	# Used estimates of rates of homicides per 100 000 population
homicides = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/VIOLENCE_HOMICIDENUM,VIOLENCE_HOMICIDERATE&profile=crosstable&filter=COUNTRY:*;AGEGROUP:-;SEX:-&x-sideaxis=COUNTRY&x-topaxis=GHO;YEAR")

# 10. Road traffic deaths (from: http://apps.who.int/gho/data/node.main.A997?lang=en)
	# Used 2013 (one year only)
	# Used estimated road traffic death rate (per 100 000 population)
traffdeath = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/RS_196,RS_198&profile=crosstable&filter=COUNTRY:*&x-sideaxis=COUNTRY&x-topaxis=GHO;YEAR")

# 11. Tuberculosis (from: http://apps.who.int/gho/data/view.main.57040ALL?lang=en)
	# Used 2014
	# Used all cases
	# Used incidence
tb = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/MDG_0000000020,TB_e_inc_num,TB_e_inc_tbhiv_100k,TB_e_inc_tbhiv_num&profile=crosstable&filter=COUNTRY:*;REGION:*&x-sideaxis=COUNTRY;YEAR&x-topaxis=GHO")

# 12. Suicide (from: http://apps.who.int/gho/data/node.main.MHSUICIDE?lang=en)
	# Used 2012
	# Used both sexes
	# Used age-standardized suicide rates
suicide = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/MH_12&profile=crosstable&filter=COUNTRY:*;REGION:*&x-sideaxis=COUNTRY&x-topaxis=GHO;YEAR;SEX")

# Create function for cleaning each imported DataFrame
def cleaningData(data, rowsToKeep, outcome, colsToDrop = [], varNames = [], colsToConvert = [], year = None):
    d = data.ix[rowsToKeep : ]
    if colsToDrop:
        d = d.drop(d.columns[colsToDrop], axis = 1)
    
    d.columns = varNames
    
    if (d[outcome].dtype == 'O'):
        if (d[outcome].str.contains("\[").any()):
            d[outcome] = d[outcome].apply(lambda x: x.split(' [')[0])
            d[outcome] = d[outcome].str.replace(' ', '')
    
    d[colsToConvert] = d[colsToConvert].apply(lambda x: pd.to_numeric(x, errors ='coerce'))
    
    if 'Year' in list(d.columns.values):
        d = d.loc[d['Year'] == year]
        del d['Year']
    return d

# Clean each DataFrame below. Note that 'alcohol' required 2 additional cleaning steps as it
# had a very different sturcture from the other data sources.

life = cleaningData(life, 1, 'LifeExpectancy', range(3, 8), ['Country', 'Year', 'LifeExpectancy'],
					['Year', 'LifeExpectancy'], 2015)

oweight = cleaningData(oweight, 3, 'Overweight', range(2, 7), ['Country', 'Overweight'], ['Overweight'])

chol = cleaningData(chol, 1, 'Cholesterol', [2, 4, 5, 6, 7, 8], ['Country', 'Year', 'Cholesterol'],
                    ['Year', 'Cholesterol'], 2008)

bsugar = cleaningData(bsugar, 1, 'BloodSugar', [2, 4, 5, 6, 7], ['Country', 'Year', 'BloodSugar'],
                    ['Year', 'BloodSugar'], 2014)

sanitation = cleaningData(sanitation, 1, 'ImprovedWater', [2, 3, 5, 6], 
                           ['Country', 'Year', 'ImprovedWater', 'ImprovedSanitation'],
                           ['Year', 'ImprovedWater', 'ImprovedSanitation'], 2015)

maternal = cleaningData(maternal, 0, 'MaternalDeaths', [3], ['Country', 'Year', 'MaternalDeaths'],
                           ['Year', 'MaternalDeaths'], 2015)

uvrad = cleaningData(uvrad, 1, 'UVRadiation', [], ['Country', 'UVRadiation'], 
                      ['UVRadiation'])

homicides = cleaningData(homicides, 1, 'HomicideRate', [1], ['Country', 'HomicideRate'],
                    ['HomicideRate'])

traffdeath = cleaningData(traffdeath, 1, 'TrafficDeaths', [1], ['Country', 'TrafficDeaths'],
                    ['TrafficDeaths'])

tb = cleaningData(tb, 0, 'Tubercululosis', [2, 4, 5], ['Country', 'Year', 'Tubercululosis'],
                    ['Year', 'Tubercululosis'], 2014)

suicide = cleaningData(suicide, 2, 'Suicide', [2, 3], ['Country', 'Suicide'],
                    ['Suicide'])


alcohol = cleaningData(alcohol, 1, 'Alcohol', [1] + range(3, 8) + range(9, 19), ['Country', 'Type', 'Alcohol'],
                    ['Alcohol'])
alcohol = alcohol[alcohol['Type'].str.contains("All types")]
del alcohol['Type']

# Now that the data is cleaned, time to merge all of the variables into one DataFrame.
def mergeFunc(dataframe1, dataframe2):
    return pd.merge(dataframe1, dataframe2, left_on='Country', 
                    right_on='Country', how='outer')

totaldf = mergeFunc(life, alcohol)
for i in [oweight, chol, bsugar, sanitation, maternal,
         uvrad, homicides, traffdeath, tb, suicide]:
    totaldf = mergeFunc(totaldf, i)

# Delete all rows with missing values from full DataFrame
totaldf = totaldf.dropna()

# As LifeExpectancy is negatively skewed, I will apply an inverse square root transformation.
totaldf['TransformedLife'] = np.sqrt((max(totaldf['LifeExpectancy']) + 1) - totaldf['LifeExpectancy'])

# LASSO regression was the most appropriate technique (when compared to ridge regression).
# Using evaluation techniques and code from http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/

# First, standardise the predictors
for i in list(totaldf.columns.values)[2:14]:
    totaldf['%s' % i] = (totaldf['%s' % i] - totaldf['%s' % i].mean()) / totaldf['%s' %i].std()

# Then assess the different alphas for the LASSO regression:
def lasso_regression(data, predictors, alpha):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['TransformedLife'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['TransformedLife'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

predictors = list(totaldf.columns.values)[2:14]

alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

col = ['rss','intercept'] + list(totaldf.columns.values)[2:14]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(totaldf, predictors, alpha_lasso[i])

coef_matrix_lasso

# The LASSO indicates that the model with the greatest parsimony but the lowest RSS has
# an alpha of 0.001
lassoreg = Lasso(alpha=0.001, normalize=True, max_iter=1e5)
lassoreg.fit(totaldf[list(totaldf.columns.values)[2:14]], totaldf['TransformedLife'])
rss = sum((y_pred-totaldf['TransformedLife'])**2)
ret = [rss]
ret.extend([lassoreg.intercept_])
ret.extend(lassoreg.coef_)

# Print out the coefficients of the model:
print("Model coefficients")
print({key:value for key, value in zip(list(totaldf.columns.values)[2:14], [round(elem, 2) for elem in ret])})

# RSS:
print("\nModel RSS")
print(sum((y_pred-totaldf['TransformedLife'])**2))

# Predicted life expectancy on same data
print("\nPredicted values by country")
print({key:value for key, value in zip(list(totaldf['Country']),
                                 [round(elem, 1) for elem in (max(totaldf['LifeExpectancy']) + 1) - (y_pred**2)])})
