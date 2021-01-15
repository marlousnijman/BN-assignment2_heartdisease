### Install Packages
# install.packages('remotes', dependencies = TRUE)
# install.packages('bnlearn', dependencies=TRUE)
# install.packages('pROC', dependencies=TRUE)
# install.packages('ggplot2', dependencies=TRUE)
# install.packages('caret', dependencies=TRUE)
# install.packages("dagitty", dependencies = TRUE)
# remotes::install_github("jtextor/bayesianNetworks")

### Libraries
library(dagitty)
library(bayesianNetworks)
library(bnlearn)
library(pROC)
library(ggplot2)
library(caret)


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


### Tabu Search
from <- rep("diagnosis", 13)
to <- c("age", "sex", "chest_pain", "rest_blood_press", 
        "cholesterol", "fasting_blood_sugar", "rest_ecg", 
        "max_heart_rate", "exercise_induced_angina", 
        "ST_depression", "ST_slope", "coloured_arteries",
        "thalassemia")

blacklist <- data.frame(from = from, to = to); blacklist

tabu_net <- tabu(data, maxp = 4, blacklist = blacklist)
par(cex=0.9)
plot(tabu_net)


### Evaluation Metric
# Convert to bn
pruned_net_bn <- model2network(toString(pruned_net,"bnlearn")) 

# Compute Structural Hamming Distance
shd(tabu_net, pruned_net_bn)


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
  fit <- bn.fit(tabu_net, train_data); fit
  
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
png("plots/tabu_net_confusion_matrix.png", width = 650)
ggplot(data = as.data.frame(cm$table), aes(sort(Reference,decreasing = T), Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq), size = 7) +
  scale_fill_gradient(low="white", high="#B4261A") +
  labs(x = "Ground Truth",y = "Prediction", fill="Frequency", title = "Pruned Bayesian Network Predictions", size=8) +
  scale_x_discrete(labels=c("Heart Disease", "No Heart Disease")) +
  scale_y_discrete(labels=c("No Heart Disease", "Heart Disease")) +
  theme_bw(base_size = 15)
dev.off()


### Plot network using Dagitty
tabu_net_dag <- dagitty('dag {
bb="0,0,1,1"
ST_depression [pos="0.694,0.155"]
ST_slope [pos="0.700,0.454"]
age [pos="0.066,0.474"]
chest_pain [pos="0.334,0.277"]
cholesterol [pos="0.491,0.586"]
coloured_arteries [pos="0.703,0.791"]
diagnosis [pos="0.961,0.424"]
exercise_induced_angina [pos="0.280,0.595"]
fasting_blood_sugar [pos="0.617,0.899"]
max_heart_rate [pos="0.530,0.232"]
rest_blood_press [pos="0.526,0.410"]
rest_ecg [pos="0.516,0.132"]
sex [pos="0.114,0.186"]
thalassemia [pos="0.115,0.834"]
 exercise_induced_angina -> ST_depression [beta = " 0.27 "]
 ST_depression -> ST_slope [beta = " 0.59 "]
 ST_slope -> age [beta = " 0.13 "]
 exercise_induced_angina -> chest_pain [beta = " 0.38 "]
 sex -> cholesterol [beta = " -0.16 "]
 age -> coloured_arteries [beta = " 0.33 "]
 chest_pain -> coloured_arteries [beta = " 0.17 "]
 thalassemia -> coloured_arteries [beta = " 0.18 "]
 chest_pain -> diagnosis [beta = " 0.21 "]
 coloured_arteries -> diagnosis [beta = " 0.37 "]
 exercise_induced_angina -> diagnosis [beta = " 0.25 "]
 thalassemia -> diagnosis [beta = " 0.38 "]
 rest_blood_press -> fasting_blood_sugar [beta = " 0.17 "]
 ST_slope -> max_heart_rate [beta = " -0.24 "]
 age -> max_heart_rate [beta = " -0.32 "]
 chest_pain -> max_heart_rate [beta = " -0.18 "]
 exercise_induced_angina -> max_heart_rate [beta = " -0.18 "]
 age -> rest_blood_press [beta = " 0.27 "]
 rest_blood_press -> rest_ecg [beta = " 0.16 "]
 age -> sex [beta = " -0.09 "]
 exercise_induced_angina -> thalassemia [beta = " 0.33 "]
}

')

### Determine Edge Coefficients
edges = ""
for( x in names(tabu_net_dag) ){
  px <- dagitty::parents(tabu_net_dag, x)
  for( y in px ){
    tst <- ci.test( x, y,setdiff(px,y), data=data )
    
    # Print edges
    print(paste(y,'->',x, tst$statistic, tst$p.value ) )
    
    # Determine edge coefficients
    edges <- paste(edges,y,'->',x, '[beta = "',round(tst$statistic, digits = 2),'"]\n')
  }
}
cat(edges)


### Plot Network
png('plots/tabu_net.png', width = 750, height = 750)
plot(tabu_net_dag, show.coefficients=TRUE)
dev.off()