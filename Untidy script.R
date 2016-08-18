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

head(exercise)
exercise$PhysicalActivity <- gsub("\\[.*", "", exercise$PhysicalActivity)
exercise$PhysicalActivity <- as.numeric(exercise$PhysicalActivity)
class(exercise$PhysicalActivity)

head(chol)
chol$Cholesterol <- gsub("\\[.*", "", chol$Cholesterol)
chol$Cholesterol <- as.numeric(chol$Cholesterol)
class(chol$Cholesterol)

head(bsugar)
bsugar$BloodSugar <- gsub("\\[.*", "", bsugar$BloodSugar)
bsugar$BloodSugar <- as.numeric(bsugar$BloodSugar)
class(bsugar$BloodSugar)

head(uvrad)
class(uvrad$UVRadiation)

head(water)
class(water$ImprovedWater)

head(sanitation)
class(sanitation$ImprovedSanitation)

head(maternal)
maternal$MaternalDeaths <- gsub("\\[.*", "", maternal$MaternalDeaths)
maternal$MaternalDeaths <- as.numeric(maternal$MaternalDeaths)
class(maternal$MaternalDeaths)

head(homicides)
homicides$Homicides <- gsub("\\[.*", "", homicides$Homicides)
homicides$Homicides <- as.numeric(homicides$Homicides)
class(homicides$Homicides)

head(traffdeath)
class(traffdeath$RoadDeaths)

head(malaria)
malaria$Malaria <- gsub(" |&lt;", "", malaria$Malaria)
malaria$Malaria <- gsub("\\[.*", "", malaria$Malaria)
malaria$Malaria <- as.numeric(malaria$Malaria)
class(malaria$Malaria)

head(hiv)
hiv$HIV <- gsub(" |&lt;", "", hiv$HIV)
hiv$HIV <- gsub("\\[.*", "", hiv$HIV)
hiv$HIV <- as.numeric(hiv$HIV)
class(hiv$HIV)

head(tb)
tb$Tuberculosis <- gsub("\\[.*", "", tb$Tuberculosis)
tb$Tuberculosis <- as.numeric(tb$Tuberculosis)
class(tb$Tuberculosis)

head(suicide)
class(suicide$Suicide)

