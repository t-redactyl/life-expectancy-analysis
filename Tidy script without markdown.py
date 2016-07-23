# Import all necessary libraries
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.linear_model import Lasso

# Import life expectancy and predictor data from World Health Organisation Global
# Health Observatory data repo (http://www.who.int/gho/en/)

# Make note in presentation that I already know that Smoking, Physical Activity, Malaria and HIV
# have too many missing values to include, so I won't even bother importing them.

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
alcohol = dataImport("http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/SA_0000001400&profile=crosstable&filter=COUNTRY:*;YEAR:2015;YEAR:2x014;YEAR:2013;YEAR:2012;YEAR:2011;YEAR:2010;YEAR:2009;YEAR:2008;YEAR:2007;YEAR:2006;YEAR:2005;YEAR:2004;YEAR:2003;YEAR:2002;YEAR:2001;YEAR:2000&x-sideaxis=COUNTRY;DATASOURCE;ALCOHOLTYPE&x-topaxis=GHO;YEAR")

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