# Import all necessary libraries
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import statistics as stats

# Import life expectancy and predictor data from World Health Organisation Global
# Health Observatory data repo (http://www.who.int/gho/en/)

# 1. Life expectancy
url = "http://apps.who.int/gho/athena/data/xmart.csv?target=GHO/WHOSIS_000001,WHOSIS_000015&profile=crosstable&filter=COUNTRY:*&x-sideaxis=COUNTRY;YEAR&x-topaxis=GHO;SEX"
life = pd.read_csv(url) 

