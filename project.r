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

### Pruned Network
pruned_net <- dagitty('dag {
bb="-5.198,-4.944,6.567,7.657"
ST_depression [pos="-3.579,-4.576"]
ST_slope [pos="-4.217,-1.331"]
age [pos="2.272,0.152"]
chest_pain [pos="-0.885,-0.152"]
cholesterol [pos="1.525,3.714"]
coloured_arteries [pos="1.210,-2.529"]
diagnosis [pos="-1.711,-3.396"]
exercise_induced_angina [pos="-4.263,1.062"]
fasting_blood_sugar [pos="3.897,-2.776"]
max_heart_rate [pos="-2.097,-1.628"]
rest_blood_press [pos="5.587,-0.105"]
rest_ecg [pos="4.512,2.476"]
sex [pos="-0.695,2.591"]
thalassemia [pos="-2.788,1.952"]
ST_depression -> diagnosis [beta=" 0.13 "]
ST_slope -> ST_depression [beta=" 0.54 "]
ST_slope -> diagnosis [beta=" 0.15 "]
age -> cholesterol [beta=" 0.13 "]
age -> coloured_arteries [beta=" 0.34 "]
age -> fasting_blood_sugar [beta=" 0.12 "]
age -> max_heart_rate [beta=" -0.34 "]
age -> rest_blood_press [beta=" 0.27 "]
age -> rest_ecg [beta=" 0.14 "]
chest_pain -> coloured_arteries [beta=" 0.23 "]
chest_pain -> diagnosis [beta=" 0.29 "]
chest_pain -> max_heart_rate [beta=" -0.19 "]
coloured_arteries -> diagnosis [beta=" 0.36 "]
exercise_induced_angina -> chest_pain [beta=" 0.33 "]
exercise_induced_angina -> max_heart_rate [beta=" -0.23 "]
fasting_blood_sugar -> coloured_arteries [beta=" 0.12 "]
max_heart_rate -> ST_depression [beta=" -0.17 "]
max_heart_rate -> ST_slope [beta=" -0.28 "]
max_heart_rate -> diagnosis [beta=" -0.15 "]
sex -> cholesterol [beta=" -0.15 "]
thalassemia -> ST_slope [beta=" 0.23 "]
thalassemia -> chest_pain [beta=" 0.16 "]
thalassemia -> exercise_induced_angina [beta=" 0.33 "]
}
')


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
head(data)

### pc-algorithm
nlev <- as.vector(sapply(sapply(data, unique), length))
labels <- colnames(data)
suffStat <- list(dm = data, nlev = nlev, adaptDF = FALSE)
pc.fit = pc(suffStat = suffStat, indepTest = disCItest, alpha = 0.05, labels = labels, m.max = 1)
par(cex=0.5)
plot(pc.fit)
 
### Evaluation Metric

# Using NetworkDistance
# pc algorithm network adjacancy matrix
adj_mat_pc <- as.table(as(pc.fit, 'amat'))

# original network adjacancy matrix
pruned_net_bn <- model2network(toString(pruned_net,"bnlearn")) 
adj_mat_bn <- amat(pruned_net_bn)

input <-list(adj_mat_pc, adj_mat_bn)
nd.hamming(A = input)

# Using bnlearn
pc_net_bn <- as.bn(pc.fit)
shd(pc_net_bn, pruned_net_bn)


### Cross Validation
k = 10
folds = createFolds(data$sex, k = k)
all_preds <- NULL
all_labels <- NULL
for (test_index in folds) {
  # Split data into test and train
  train_index <- setdiff(1:nrow(data), test_index)
  test_data = data[test_index,]
  train_data = data[train_index,]
  
  # Fit on data
  fit <- bn.fit(pc_net_bn, train_data); fit
  
  # Predict 
  preds <- predict(fit, node= 'diagnosis', data = test_data, method = "bayes-lw", n = 10000) 
  
  # Save all data
  all_preds <- c(all_preds, preds)
  all_labels <- c(all_labels, test_data$diagnosis)
}

### Analysis

# Check range
range(all_preds)

# Round values
all_preds = round(all_preds)

# Confusion Matrix
cm <- confusionMatrix(data = factor(all_preds), reference = factor(all_labels)); cm
png("plots/pc_net_confusion_matrix.png", width = 650)
ggplot(data = as.data.frame(cm$table), aes(sort(Reference,decreasing = T), Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq), size = 7) +
  scale_fill_gradient(low="white", high="#B4261A") +
  labs(x = "Ground Truth",y = "Prediction", fill="Frequency", title = "Bayesian Network Predictions", size=8) +
  scale_x_discrete(labels=c("Heart Disease", "No Heart Disease")) +
  scale_y_discrete(labels=c("No Heart Disease", "Heart Disease")) +
  theme_bw(base_size = 15)
dev.off()