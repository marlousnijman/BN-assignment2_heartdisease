### Install Packages
# install.packages('remotes', dependencies = TRUE)
# install.packages('bnlearn', dependencies=TRUE)
# install.packages('pROC', dependencies=TRUE)
# install.packages('ggplot2', dependencies=TRUE)
# install.packages('caret', dependencies=TRUE)
# install.packages("dagitty", dependencies = TRUE)
# install.packages('pcalg', dependencies=TRUE)
# install.packages("BiocManager")
# install.packages('NetworkDistance', dependencies=TRUE)
# BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
# remotes::install_github("jtextor/bayesianNetworks")

### Libraries
library(dagitty)
library(bayesianNetworks)
library(bnlearn)
library(pROC)
library(ggplot2)
library(caret)
library(pcalg)
library(NetworkDistance)


### Data
data <- read.csv("data/processed_cleveland.csv", header = FALSE)
colnames(data) <- c("age", "sex", "chest_pain", "rest_blood_press", 
                    "cholesterol", "fasting_blood_sugar", "rest_ecg", 
                    "max_heart_rate", "exercise_induced_angina", 
                    "ST_depression", "ST_slope", "coloured_arteries",
                    "thalassemia", "diagnosis")
head(data)


### Preprocessing

# NANs
nrow(data[which(data$coloured_arteries == '?'),])
nrow(data[which(data$thalassemia == '?'),])

# Set these to values that occur most in the dataset 
counts_thal <- table(data$thalassemia)
barplot(counts_thal) # The most occuring value is 3.0

counts_col <- table(data$coloured_arteries)
barplot(counts_col) # The most occuring value is 0.0

data$coloured_arteries[which(data$coloured_arteries == '?')] <- '0.0'
data$thalassemia[which(data$thalassemia == '?')] <- '3.0'

# Convert to numeric
data$thalassemia <- as.numeric(data$thalassemia)
data$coloured_arteries <- as.numeric(data$coloured_arteries)
data$diagnosis <- as.numeric(data$diagnosis)

### Dealing with different types of data

# Convert continuous data to categorical data
data$age <- as.numeric(cut(data$age, 5))
data$rest_blood_press <- as.numeric(cut(data$rest_blood_press, c(90, 120, 140, 200), labels = c(1,2,3)))
data$cholesterol <- as.numeric(cut(data$cholesterol, c(100, 200, 300, 600), labels = c(1,2,3)))
data$max_heart_rate <- as.numeric(cut(data$max_heart_rate, c(50, 110, 140, 175, 210), labels = c(1,2,3,4)))
data$ST_depression <- as.numeric(cut(data$ST_depression, c(-0.1, 0.0, 2, 6.5), labels = c(0,1,2)))

# Bin diagnosis
data$diagnosis[which(data$diagnosis > 0)] <- 1 


### pc-algorithm
nlev <- as.vector(sapply(sapply(data, unique), length))
labels <- colnames(data)
suffStat <- list(dm = data, nlev = nlev, adaptDF = FALSE)
pc.fit = pc(suffStat = suffStat, indepTest = disCItest, alpha = 0.05, labels = labels, m.max = 3)
par(cex=0.5)
plot(pc.fit)


### Tabu algorithm
from <- rep("diagnosis", 13)
to <- c("age", "sex", "chest_pain", "rest_blood_press", 
        "cholesterol", "fasting_blood_sugar", "rest_ecg", 
        "max_heart_rate", "exercise_induced_angina", 
        "ST_depression", "ST_slope", "coloured_arteries",
        "thalassemia")

blacklist <- data.frame(from = from, to = to); blacklist

tabu_net <- tabu(data, maxp = 4, blacklist = blacklist)

### Evaluation Metric
# Convert to bn
pc_net_bn <- as.bn(pc.fit)

# Compute Structural Hamming Distance
shd(pc_net_bn, tabu_net)
























