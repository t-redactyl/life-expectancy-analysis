# Load in the data
path <- "/Users/jburchell/Dropbox/Python projects/Life expectancy analysis/"
life <- read.csv(paste0(path, "Life expectancy_final year.csv"))
smoking <- read.csv(paste0(path, "Smoking_final year.csv"))
alcohol <- read.csv(paste0(path, "Alcohol_final year.csv"))
oweight <- read.csv(paste0(path, "Overweight_final year.csv"))
exercise <- read.csv(paste0(path, "Physical activity_final year.csv"))
chol <- read.csv(paste0(path, "Cholesterol_final year.csv"))
bsugar <- read.csv(paste0(path, "Blood sugar_final year.csv"))
water <- read.csv(paste0(path, "Water_final year.csv"))
sanitation <- read.csv(paste0(path, "Sanitation_final year.csv"))
maternal <- read.csv(paste0(path, "Maternal death_final year.csv"))
uvrad <- read.csv(paste0(path, "UV radiation_final years.csv"))
homicides <- read.csv(paste0(path, "Homicide_final year.csv"))
traffdeath <- read.csv(paste0(path, "Road traffic deaths_final year.csv"))
malaria <- read.csv(paste0(path, "Malaria_final year.csv"))
hiv <- read.csv(paste0(path, "HIV_final year.csv"))
tb <- read.csv(paste0(path, "Tuberculosis_final year.csv"))
suicide <- read.csv(paste0(path, "Suicide_final year.csv"))

# Screening the data
head(life)
class(life$LifeExpectancy)

head(smoking)
class(smoking$Smoking)
smoking$Smoking <- as.numeric(as.character(smoking$Smoking))
class(smoking$Smoking)

head(alcohol)
class(alcohol$Alcohol)
alcohol$Alcohol <- as.numeric(as.character(alcohol$Alcohol))
class(alcohol$Alcohol)

#library(stringr)
head(oweight)
oweight$Overweight <- gsub("\\[.*", "", oweight$Overweight)
oweight$Overweight <- as.numeric(oweight$Overweight)
class(oweight)

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