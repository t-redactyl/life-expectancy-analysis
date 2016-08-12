import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import statistics as stats

# Load in the data
path = "/Users/jburchell/Dropbox/Python projects/Life expectancy analysis/"
life = pd.read_csv("%sLife expectancy_final year.csv" % path)
smoking = pd.read_csv("%sSmoking_final year.csv" % path)
alcohol = pd.read_csv("%sAlcohol_final year.csv" % path)
oweight = pd.read_csv("%sOverweight_final year.csv" % path)
exercise = pd.read_csv("%sPhysical activity_final year.csv" % path)
chol = pd.read_csv("%sCholesterol_final year.csv" % path)
bsugar = pd.read_csv("%sBlood sugar_final year.csv" % path)
water = pd.read_csv("%sWater_final year.csv" % path)
sanitation = pd.read_csv("%sSanitation_final year.csv" % path)
maternal = pd.read_csv("%sMaternal death_final year.csv" % path)
uvrad = pd.read_csv("%sUV radiation_final years.csv" % path)
homicides = pd.read_csv("%sHomicide_final year.csv" % path)
traffdeath = pd.read_csv("%sRoad traffic deaths_final year.csv" % path)
malaria = pd.read_csv("%sMalaria_final year.csv" % path)
hiv = pd.read_csv("%sHIV_final year.csv" % path)
tb = pd.read_csv("%sTuberculosis_final year.csv" % path)
suicide = pd.read_csv("%sSuicide_final year.csv" % path)

# Screening the data
life[ :5]
life['LifeExpectancy'].dtype

smoking[ :5]
smoking['Smoking'].dtype
smoking['Smoking'] = smoking['Smoking'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
smoking['Smoking'].dtype

alcohol[ :5]
alcohol['Alcohol'].dtype
alcohol['Alcohol'] = alcohol['Alcohol'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
alcohol['Alcohol'].dtype

oweight[ :5]
oweight['Overweight'] = oweight['Overweight'].apply(lambda x: x.split(' [')[0])
oweight['Overweight'] = oweight['Overweight'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
oweight['Overweight'].dtype

exercise[ :5]
exercise['PhysicalActivity'].dtype
exercise['PhysicalActivity'] = exercise['PhysicalActivity'].apply(lambda x: x.split(' [')[0])
exercise['PhysicalActivity'] = exercise['PhysicalActivity'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
exercise['PhysicalActivity'].dtype

chol[ :5]
chol['Cholesterol'] = chol['Cholesterol'].apply(lambda x: x.split(' [')[0])
chol['Cholesterol'] = chol['Cholesterol'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
chol['Cholesterol'].dtype

bsugar[ :5]
bsugar['BloodSugar'] = bsugar['BloodSugar'].apply(lambda x: x.split(' [')[0])
bsugar['BloodSugar'] = bsugar['BloodSugar'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
bsugar['BloodSugar'].dtype

uvrad[ :5]
uvrad['UVRadiation'].dtype

water[ :5]
water['ImprovedWater'].dtype

sanitation[ :5]
sanitation['ImprovedSanitation'].dtype

maternal[ :5]
maternal['MaternalDeaths'] = maternal['MaternalDeaths'].apply(lambda x: x.split(' [')[0])
maternal['MaternalDeaths'] = maternal['MaternalDeaths'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
maternal['MaternalDeaths'].dtype

homicides[ :5]
homicides['Homicides'] = homicides['Homicides'].apply(lambda x: x.split(' [')[0])
homicides['Homicides'] = homicides['Homicides'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
homicides['Homicides'].dtype

traffdeath[ :5]
traffdeath['RoadDeaths'].dtype

malaria[ :5]
malaria['Malaria'] = malaria['Malaria'].str.replace(' ', '')
malaria['Malaria'] = malaria['Malaria'].str.replace('&lt;', '')
malaria['Malaria'] = malaria['Malaria'].apply(lambda x: x.split('[')[0])
malaria['Malaria'] = malaria['Malaria'].apply(lambda x: pd.to_numeric(x, errors ='coerce'))
malaria['Malaria'].dtype

hiv[ :5]
hiv['HIV'] = hiv['HIV'].str.replace(' ', '')
hiv['HIV'] = hiv['HIV'].str.replace('&lt;', '')
hiv['HIV'] = hiv['HIV'].apply(lambda x: x.split('[')[0])
hiv['HIV'] = hiv['HIV'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 

tb[ :5]
tb['Tuberculosis'] = tb['Tuberculosis'].apply(lambda x: x.split(' [')[0])
tb['Tuberculosis'] = tb['Tuberculosis'].apply(lambda x: pd.to_numeric(x, errors ='coerce')) 
tb['Tuberculosis'].dtype

suicide[ :5]
suicide['Suicide'].dtype

# Merge the data
def mergeFunc(dataframe1, dataframe2):
    return pd.merge(dataframe1, dataframe2, left_on='Country', 
                    right_on='Country', how='outer')

totaldf = mergeFunc(life, smoking)
for i in [alcohol, oweight, exercise, chol, bsugar, water, sanitation, maternal,
         uvrad, homicides, traffdeath, malaria, hiv, tb, suicide]:
    totaldf = mergeFunc(totaldf, i)
totaldf[ :5]

# Check for missingness
totaldf.isnull().sum()
	# Lots of missing data for Smoking, Physical Activity, Malaria and HIV, so won't use
	# 14 missing values in Life Expectancy so will delete them
	# Will delete all non-complete rows once I get rid of the above 4 columns

totaldf = totaldf.drop(['Smoking', 'PhysicalActivity', 'Malaria', 'HIV'], axis=1) 
totaldf = totaldf.dropna()
totaldf.isnull().sum()
totaldf.shape

# Explore the data a bit
totaldf.ix[totaldf['AlcConsumption'].idxmax()]
totaldf.ix[totaldf['ImprovedWater'].idxmin()]

Series.mean(totaldf['Suicide'])
print(totaldf.loc[totaldf['Country'].isin(['Panama', 'Guatemala', 'Australia'])])

# Work out if can use linear regression
corr_df = totaldf.corr(method='pearson')
print("--------------- CORRELATIONS ---------------")
print(corr_df.head(corr_df.shape[1]))

s = corr_df.unstack()
so = DataFrame(s.sort_values(kind="quicksort"))
so.loc[(so[0] >= .8) & (so[0] < 1)]

import matplotlib.pyplot as plt

plt.hist(totaldf['LifeExpectancy'])
plt.title("Life Expectancy Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
	# Needs transformation

import math
#math.sqrt(totaldf['LifeExpectancy'])

plt.hist(np.sqrt((max(totaldf['LifeExpectancy']) + 1) - totaldf['LifeExpectancy']))
plt.title("Life Expectancy Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Check back-transform
(max(totaldf['LifeExpectancy']) + 1) - (totaldf['TransformedLife']**2)[:5]

# Create the variable
totaldf['TransformedLife'] = np.sqrt((max(totaldf['LifeExpectancy']) + 1) - totaldf['LifeExpectancy'])

# Standardise the predictors
for i in list(totaldf.columns.values)[2:14]:
    totaldf['%s' % i] = (totaldf['%s' % i] - totaldf['%s' % i].mean()) / totaldf['%s' %i].std()

# Try out the ridge/LASSO table of results from blog post
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')

for i in range(2,16):  
    colname = 'x_%d'%i 
    data[colname] = data['x']**i
print data.head()

from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)

pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge

# Looks good, let's adapt for choosing regularised regression model
	# Ridge regression
from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['TransformedLife'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['TransformedLife'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

predictors = list(totaldf.columns.values)[2:14]
predictors

alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

col = ['rss','intercept'] + list(totaldf.columns.values)[2:14]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(totaldf, predictors, alpha_ridge[i])

pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge

	# LASSO regression
from sklearn.linear_model import Lasso
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

# Predicted values (same data) - intercept or no intercept
# No intercept:
#E(Y|X) = 0 + 0.053(Alcohol) - 0.36(Cholesterol) + 0.22(BloodSugar) - 0.067(ImprovedWater) - 0.22(ImprovedSanitation) +
#         0.27(MaternalDeaths) + 0.00056(UvRadiation) + 0.011(Homicides) + 0.26(RoadDeaths) + 0.23(Tuberculosis) +
#         0.047(Suicide)
        
# Intercept
#E(Y|X) = 3.4 + 0.045(Alcohol) - 0.36(Cholesterol) + 0.22(BloodSugar) - 0.065(ImprovedWater) - 0.22(ImprovedSanitation)
#         + 0.26(MaternalDeaths) + 0.01(Homicides) + 0.26(RoadDeaths) + 0.23(Tuberculosis) + 0.045(Suicide)

# Checking the best model
totaldf['PredLifeNoIntercept'] = (0 + 0.053*totaldf['Alcohol'] - 0.36*totaldf['Cholesterol'] +
                                  0.22*totaldf['BloodSugar'] - 0.067*totaldf['ImprovedWater'] - 
                                  0.22*totaldf['ImprovedSanitation'] + 0.27*totaldf['MaternalDeaths'] +
                                  0.00056*totaldf['UVRadiation'] + 0.011*totaldf['Homicides'] + 
                                  0.26*totaldf['RoadDeaths'] + 0.23*totaldf['Tuberculosis'] +
                                  0.047*totaldf['Suicide'])
totaldf['PredLifeNoIntercept'] = (max(totaldf['LifeExpectancy']) + 1) - (totaldf['PredLifeNoIntercept']**2)

totaldf['PredLifeIntercept'] = (3.4 + 0.045*totaldf['Alcohol'] - 0.36*totaldf['Cholesterol'] + 
                                0.22*totaldf['BloodSugar'] - 0.065*totaldf['ImprovedWater'] - 
                                0.22*totaldf['ImprovedSanitation'] + 0.26*totaldf['MaternalDeaths'] + 
                                0.01*totaldf['Homicides'] + 0.26*totaldf['RoadDeaths'] + 
                                0.23*totaldf['Tuberculosis'] + 0.045*totaldf['Suicide'])
totaldf['PredLifeIntercept'] = (max(totaldf['LifeExpectancy']) + 1) - (totaldf['PredLifeIntercept']**2)

# -*- end -*-
