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




### Data Inspection

# Continuous Variables
range(data$age)
range(data$rest_blood_press)
range(data$cholesterol)
range(data$max_heart_rate)
range(data$ST_depression)

# Categorical Variables
factor(data$sex)[1]
factor(data$chest_pain)[1]
factor(data$fasting_blood_sugar)[1]
factor(data$rest_ecg)[1]
factor(data$exercise_induced_angina)[1]
factor(data$ST_slope)[1]
factor(data$coloured_arteries)[1] 
factor(data$thalassemia)[1] 
factor(data$diagnosis)[1] 

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



### tabu search

tabu_1 <- tabu(data)
plot(tabu_1)
tabu_1$arcs
# remove all arcs from diagnosis to *
bklst_filtered <- rbind(tabu_1$arcs[3:7,], tabu_1$arcs[15,], tabu_1$arcs[19,])
bklst_filtered

tabu_2 <- tabu(data, blacklist = bklst_filtered)
plot(tabu_2)
tabu_2$arcs
# remove all arcs from diagnosis to *
bklst_filtered_2 <- rbind(bklst_filtered, tabu_2$arcs[9,], tabu_2$arcs[17,])
bklst_filtered_2

tabu_3 <- tabu(data, blacklist = bklst_filtered_2)
plot(tabu_3)



### convert tabu_3 to dagitty structure ...









### Test Network Structure 
impliedConditionalIndependencies(tabu_3)

# Chi-squared Test (only for categorical variables)
localTests(pruned_net, data, type="cis.chisq", max.conditioning.variables = 4)

### Edge Coefficients
edges = ""
for( x in names(net) ){
  px <- dagitty::parents(net, x)
  for( y in px ){
    tst <- ci.test( x, y,setdiff(px,y), data=data )
    
    # Print edges
    print(paste(y,'->',x, tst$statistic, tst$p.value ) )
    
    # Determine edges to make the pruned net
    # if(tst$p.value < 0.05){
    # edges <- paste(edges,y,'->',x, '[beta = "',round(tst$statistic, digits = 2),'"]\n')
    # }
  }
}
cat(edges)

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
  
  # Convert model to bnlearn
  net_bn <- model2network(toString(pruned_net,"bnlearn")) 
  
  # Fit on data
  fit <- bn.fit(net_bn, train_data); fit
  
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

# ROC & AUC
png("plots/prunednet_roc.png")
plot(roc(all_preds, all_labels))
dev.off()
auc(all_preds, all_labels)

# Confusion Matrix
cm <- confusionMatrix(data = factor(all_preds), reference = factor(all_labels)); cm
png("plots/prunednet_confusion_matrix.png", width = 650)
ggplot(data = as.data.frame(cm$table), aes(sort(Reference,decreasing = T), Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq), size = 7) +
  scale_fill_gradient(low="white", high="#B4261A") +
  labs(x = "Ground Truth",y = "Prediction", fill="Frequency", title = "Pruned Bayesian Network Predictions", size=8) +
  scale_x_discrete(labels=c("Heart Disease", "No Heart Disease")) +
  scale_y_discrete(labels=c("No Heart Disease", "Heart Disease")) +
  theme_bw(base_size = 15)
dev.off()















