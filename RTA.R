# Predicting accident severity

# Data loading, library loading
install.packages("tidyr")
install.packages("rpart")
install.packages("ggplot2")
library(dummies)
library(dplyr)
library(tree)
library(randomForest)
library(tibble)
library(tidyr)
library(rpart)
library(ggplot2)
library(class)

RTA_data <- read.csv("/Users/chiaentsai/Desktop/Data Mining/project/RTA Dataset.csv")
RTA_data <- as_tibble(RTA_data)
#typeof(RTA_data)
head(RTA_data)
nrow(RTA_data)
ncol(RTA_data)

RTA_1 <- RTA_data[, c("Age_band_of_driver", "Sex_of_driver", "Driving_experience", "Type_of_vehicle", "Defect_of_vehicle", "Accident_severity")]

# Data preprocessing
#remove rows with missing values
RTA_1[RTA_1==""] <- NA
RTA_1 <- na.omit(RTA_1)
nrow(RTA_1)
RTA_1

#check column-Age_band_of_driver. Remove rows with unknown age.
unique(RTA_1$Age_band_of_driver)
RTA_1[RTA_1=="Unknown"] <- NA
RTA_1 <- na.omit(RTA_1)
nrow(RTA_1)
RTA_1[RTA_1=="Under 18"] <- "0"
RTA_1[RTA_1=="18-30"] <- "1"
RTA_1[RTA_1=="31-50"] <- "2"
RTA_1[RTA_1=="Over 51"] <- "3"
#RTA_1$Age_band_of_driver <- as.numeric(RTA_1$Age_band_of_driver)
RTA_1$Age_band_of_driver <- as.factor(RTA_1$Age_band_of_driver)

#check column-Sex_of_driver.
unique(RTA_1$Sex_of_driver)
RTA_1$Sex_of_driver <- as.factor(RTA_1$Sex_of_driver)

#check column-Driving_experience. Remove rows with unknown driving experience.
unique(RTA_1$Driving_experience)
RTA_1[RTA_1=="unknown"] <- NA
RTA_1 <- na.omit(RTA_1)
RTA_1$Driving_experience <- as.factor(RTA_1$Driving_experience)

#check column-Type_of_vehicle. Remove data with vehicle type "Pick up upto 10Q"(not clear), "Special vehicle"(not clear),"Turbo"(the number of data is too low). Clearly rename some columns.
table(RTA_1$Type_of_vehicle)
RTA_1[RTA_1=="Pick up upto 10Q"] <- NA
RTA_1[RTA_1=="Special vehicle"] <- NA
RTA_1[RTA_1=="Turbo"] <- NA
RTA_1 <- na.omit(RTA_1)

RTA_1[RTA_1=="Lorry (11?40Q)"] <- "Lorry (11-40Q)"
RTA_1[RTA_1=="Lorry (41?100Q)"] <- "Lorry (41-100Q)"
RTA_1[RTA_1=="Public (13?45 seats)"] <- "Public (13-45 seats)"
unique(RTA_1$Type_of_vehicle)
RTA_1$Type_of_vehicle <- as.factor(RTA_1$Type_of_vehicle)

#check column- Defect_of_vehicle
unique(RTA_1$Defect_of_vehicle)
RTA_1$Defect_of_vehicle <- as.factor(RTA_1$Defect_of_vehicle)

#check column- Accident_severity
unique(RTA_1$Accident_severity)
RTA_1$Accident_severity <- as.factor(RTA_1$Accident_severity)

```{r}
# bagging using 5 predictors (age, sex, driving experience, type of vehicle, defect of vehicle)

# get data, initial clean
roadtraffic2 = read.csv('/Users/vickiyang/ORIE 4740/dataset/RTA Dataset.csv', header=T, na.strings='Unknown')
roadtraffic2 <- na.omit(roadtraffic2)

# set up predictors/features
rt2 <- roadtraffic2[, colnames(roadtraffic2) %in% c('Age_band_of_driver','Sex_of_driver','Driving_experience', 'Type_of_vehicle', 'Defect_of_vehicle', 'Accident_severity')]

rt2[rt2=="Under 18"] <- "0"
rt2[rt2=="18-30"] <- "1"
rt2[rt2=="31-50"] <- "2"
rt2[rt2=="Over 51"] <- "3"

rt2$Age_band_of_driver <- as.factor(rt2$Age_band_of_driver)
rt2$Sex_of_driver <- as.factor(rt2$Sex_of_driver)
rt2$Driving_experience <- as.factor(rt2$Driving_experience)
rt2$Defect_of_vehicle <- as.factor(rt2$Defect_of_vehicle)

rt2[rt2=="Pick up upto 10Q"] <- NA
rt2[rt2=="Special vehicle"] <- NA
rt2[rt2=="Turbo"] <- NA
rt2 <- na.omit(rt2)

rt2[rt2=="Lorry (11?40Q)"] <- "Lorry (11-40Q)"
rt2[rt2=="Lorry (41?100Q)"] <- "Lorry (41-100Q)"
rt2[rt2=="Public (13?45 seats)"] <- "Public (13-45 seats)"
unique(rt2$Type_of_vehicle)
rt2$Type_of_vehicle <- as.factor(rt2$Type_of_vehicle)

rt2$Accident_severity <- as.factor(rt2$Accident_severity)

rt2[rt2==""] <- NA
rt2 <- na.omit(rt2)

# model creation and data splitting
set.seed(1)
train_ind2 <- sample(1:nrow(rt2), nrow(rt2)*4/5)
rt2.train <- rt2[train_ind2,]
rt2.test<- rt2[-train_ind2,]
bag.rt2 <-randomForest(Accident_severity~., data=rt2.train, mtry=5, importance=TRUE)
bag2.pred <-predict(bag.rt2, rt2.test)

# data analysis
table(bag2.pred, rt2.test$Accident_severity)
mean(bag2.pred == rt2.test$Accident_severity)

# graphing
MSE_vec2 <-rep(0,5)
for (i in (1:5)) {
  set.seed(1)
  rf.rt2 <-randomForest(Accident_severity~., data=rt2.train, mtry=i, importance=TRUE)
  rt2.pred <-predict(rf.rt2, rt2.test)
  MSE_vec2[i] <- mean(rt2.pred == rt2.test$Accident_severity)
}
# plot(MSE_vec2, xlab="m", ylab="Random Forest Test MSE")
minMSE2_index <-match(min(MSE_vec2),MSE_vec2)
rfminMSE.rt2 <-randomForest(Accident_severity~., data=rt2.train, mtry=minMSE2_index, importance=TRUE)
varImpPlot(rfminMSE.rt2)
accuracy <- mean(bag2.pred == rt2.test$Accident_severity)
accuracy


# model building
set.seed(1)
train_id <- sample(1:nrow(RTA_1), nrow(RTA_1)*4/5)
RTA_1.train <- RTA_1[train_id,]
RTA_1.test<- RTA_1[-train_id,]
RTA_1.train

# undersampling
balanced_RTA = NULL

RTA_1$Accident_severity =="Slight Injury"

slight_row <- RTA_1[RTA_1$Accident_severity =="Slight Injury",]
serious_row <- RTA_1[RTA_1$Accident_severity =="Serious Injury",]
fatal_row <- RTA_1[RTA_1$Accident_severity =="Fatal injury",]

nslight <- sum(RTA_1$Accident_severity == "Slight Injury") #4598
nserious <- sum(RTA_1$Accident_severity == "Serious Injury") #812
nfatal <- sum(RTA_1$Accident_severity == "Fatal injury") #81

serious_index <- sample(1:nrow(serious_row), 81)
slight_index <- sample(1:nrow(slight_row), 81)

slight <- slight_row[slight_index,]
serious <- serious_row[serious_index,]

balanced_RTA <- rbind(slight, serious, fatal_row)
balanced_RTA

set.seed(1)
train_id_2 <- sample(1:nrow(balanced_RTA), nrow(balanced_RTA)*4/5)
balanced_RTA.train <- balanced_RTA[train_id_2,]
balanced_RTA.test <- balanced_RTA[-train_id_2,]
balanced_RTA.train

#random forest (factors)
RTA_1.train

accuracy_2 <- rep(0,5)

for (i in 1:5) {
  
  forest_2 <- randomForest(Accident_severity~., 
                           data = RTA_1.train,
                           mtry = i,
                           importance = TRUE)
  
  pred_forest_2 <- predict(forest_2, RTA_1.test)
  accuracy_forest_2 <- mean(pred_forest_2 == RTA_1.test$Accident_severity)
  accuracy_2[i] <- accuracy_forest_2
  
}
plot(accuracy_2, type = "b")
best_m <- which.max(accuracy_2)
best_m


forest <- randomForest(Accident_severity ~.,
                       data = RTA_1.train,
                       mtry = 1,
                       importance = TRUE)

pred_forest <- predict(forest, RTA_1.test)
table(pred_forest, RTA_1.test$Accident_severity)
heatmap(table(pred_forest, RTA_1.test$Accident_severity))
forest$importance
accuracy <- mean(pred_forest == RTA_1.test$Accident_severity)
accuracy

varImpPlot(forest)


#random forest (factors) (balanced)
accuracy_3 <- rep(0,5)

for (i in 1:5) {
  
  forest_3 <- randomForest(Accident_severity~., 
                           data = balanced_RTA.train,
                           mtry = i,
                           importance = TRUE)
  
  pred_forest_3 <- predict(forest_3, balanced_RTA.test)
  accuracy_33 <- mean(pred_forest_3 == balanced_RTA.test$Accident_severity)
  accuracy_3[i] <- accuracy_33
  
}
plot(accuracy_3, type = "b")
best_m <- which.max(accuracy_3)
best_m


forest_b <- randomForest(Accident_severity ~.,
                         data = balanced_RTA.train,
                         mtry = 2,
                         importance = TRUE)

pred_forest_b <- predict(forest_b, balanced_RTA.test)
table(pred_forest_b, balanced_RTA.test$Accident_severity)
heatmap(table(pred_forest_b, balanced_RTA.test$Accident_severity))
forest$importance
accuracy_b <- mean(pred_forest_b == balanced_RTA.test$Accident_severity)
accuracy_b

varImpPlot(forest_b)

# decision tree(factors)   
set.seed(1)
decision_tree <- rpart(Accident_severity ~.,
                       RTA_1.train)
plot(decision_tree)
text(decision_tree, pretty=0)
summary(decision_tree)
pred_tree <- predict(decision_tree, RTA_1.test, type = "class")
table(pred_tree, RTA_1.test$Accident_severity)
test_error_3 <- mean(pred_tree == RTA_1.test$Accident_severity)
test_error_3

# undersampling
tree <- tree(Accident_severity ~ ., 
             balanced_RTA.train)

plot(tree)
text(tree, pretty=0)
summary(tree)
pred_tree2 <- predict(tree, balanced_RTA.test, type = "class")
table(pred_tree2, balanced_RTA.test$Accident_severity)
accuracy_before <- mean(pred_tree2 == balanced_RTA.test$Accident_severity)
accuracy_before

cv.RTA <- cv.tree(tree, FUN=prune.misclass)
cv.RTA

prune.RTA <- prune.misclass(tree, best=6)
plot(prune.RTA)
text(prune.RTA,pretty=0)
tree.pred_prune <- predict(prune.RTA, balanced_RTA.test, type="class")
table(tree.pred_prune, balanced_RTA.test$Accident_severity)
accuracy_after <- mean(tree.pred_prune == balanced_RTA.test$Accident_severity)
accuracy_after

heatmap(a, margins = c(15, 15))

# knn
RTA_11 <- RTA_1
RTA_11$Age_band_of_driver <- as.numeric(RTA_11$Age_band_of_driver)
RTA_11$Sex_of_driver <- as.numeric(RTA_11$Sex_of_driver)
RTA_11$Driving_experience <- as.numeric(RTA_11$Driving_experience)
RTA_11$Type_of_vehicle <- as.numeric(RTA_11$Type_of_vehicle)
RTA_11$Defect_of_vehicle <- as.numeric(RTA_11$Defect_of_vehicle)
RTA_11$Accident_severity <- as.numeric(RTA_11$Accident_severity)
RTA_11


set.seed(1)
train_id <- sample(1:nrow(RTA_11), nrow(RTA_11)*4/5)
RTA_11.train <- RTA_11[train_id,]
RTA_11.test<- RTA_11[-train_id,]


pred_knn <- knn(RTA_11.train, RTA_11.test, RTA_11.train$Accident_severity, k=1) 
table(pred_knn, RTA_11.test$Accident_severity)
mean(pred_knn == RTA_11.test$Accident_severity)

# Data preprocessing (turn categorical column into dummy appropriately)
RTA_11
sex_dummy <- dummy(RTA_11$Sex_of_driver)
type_dummy <- dummy(RTA_11$Type_of_vehicle)
defect_dummy <- dummy(RTA_11$Defect_of_vehicle)

RTA_dummy <- RTA_11
RTA_dummy <- RTA_dummy[,c(-2,-4,-5)]
RTA_dummy <- cbind(RTA_dummy, sex_dummy) 
RTA_dummy <- cbind(RTA_dummy, type_dummy) 
RTA_dummy <- cbind(RTA_dummy, defect_dummy) 
RTA_dummy <- as.tibble(RTA_dummy)
RTA_dummy$Accident_severity <- as.factor(RTA_dummy$Accident_severity)

# model building (dummy)
set.seed(1)
train_id <- sample(1:nrow(RTA_dummy), nrow(RTA_dummy)*4/5)
RTA_dummy.train <- RTA_dummy[train_id,]
RTA_dummy.test<- RTA_dummy[-train_id,]

#random forest (dummy)

forest_dummy <- randomForest(Accident_severity ~.,
                             data = RTA_dummy.train,
                             mtry = 10,
                             importance = TRUE)

pred_forest_dummy <- predict(forest_dummy, RTA_dummy.test)
table(pred_forest_dummy, RTA_dummy.test$Accident_severity)
heatmap(table(pred_forest_dummy, RTA_dummy.test$Accident_severity))
forest$importance
mean(pred_forest_dummy == RTA_dummy.test$Accident_severity)

# decision tree (dummy)
set.seed(1)
decision_tree_dummy <- rpart(Accident_severity ~.,
                             RTA_dummy.train)
plot(decision_tree_dummy)
summary(decision_tree_dummy)
pred_tree_dummy <- predict(decision_tree_dummy, RTA_dummy.test, type = "class")
table(pred_tree_dummy, RTA_dummy.test$Accident_severity)


tree_dummy <- tree(Accident_severity ~ ., 
                   RTA_dummy.train)
plot(tree_dummy) 
summary(tree_dummy)
pred_tree2_dummy <- predict(tree_dummy, RTA_dummy.test, type = "class")
table(pred_tree2_dummy, RTA_dummy.test$Accident_severity)

# knn (dummy)
pred_knn_dummy <- knn(RTA_dummy.train, RTA_dummy.test, RTA_dummy.train$Accident_severity, k=1) 
table(pred_knn_dummy, RTA_dummy.test$Accident_severity)
mean(pred_knn_dummy == RTA_dummy.test$Accident_severity)

library(caret)
library(class)
library(dplyr)
library(e1071)
library(FNN) 
library(gmodels) 
library(psych)

# processing data 
rta <- read.csv("RTA Dataset.csv")
rta_class <- rta[c("Age_band_of_driver", "Sex_of_driver", "Driving_experience", "Type_of_vehicle", "Defect_of_vehicle", "Accident_severity")]

# clean up unnecessary/unused data points
rta_class[rta_class == ""] <- NA
unique(rta_class$Age_band_of_driver)
rta_class[rta_class =="Unknown"] <- NA
unique(rta_class$Sex_of_driver)
rta_class[rta_class == "unknown"] <- NA
unique(rta_class$Type_of_vehicle)
rta_class[rta_class == "Other"] <- NA
unique(rta_class$Defect_of_vehicle)
rta_class <- na.omit(rta_class)
table(rta$Type_of_vehicle)

# dummy variale & binary columns & factoring 
rta_class$Sex_of_driver <- ifelse(rta_class$Sex_of_driver == "Male", 0, 1)
rta_class$Age_band_of_driver <- dummy.code(rta_class$Age_band_of_driver)
rta_class$Driving_experience <- dummy.code(rta_class$Driving_experience)
rta_class$Sex_of_driver <- dummy.code(rta_class$Sex_of_driver)
rta_class$Type_of_vehicle <- dummy.code(rta_class$Type_of_vehicle)
rta_class$Defect_of_vehicle <- dummy.code(rta_class$Defect_of_vehicle)
rta_class$Accident_severity <- as.factor(rta_class$Accident_severity)
> str(rta_class)

# splitting dataset
set.seed(123)
train_index <- sample(seq_len(nrow(rta_class)), size=floor(0.8*nrow(rta_class)))
rta_train <- rta_class[train_index, ]
rta_test <- rta_class[-train_index, ]
classifierR = svm(formula = Accident_severity ~ .,
                  data = rta_train,
                  type = 'C-classification', kernel = 'radial')
classifierL = svm(formula = Accident_severity ~ .,
                  data = rta_train,
                  type = 'C-classification',
                  kernel = 'linear')
predR = predict(classifierR, newdata=rta_test[-6])
predL = predict(classifierL, newdata=rta_test[-6])
predL = predict(classifierL, newdata=rta_test[-6])
confusionMatrixR = table(rta_test[, 6], predR)
confusionMatrixR
predR

confusionMatrixL = table(rta_test[, 6], predL)
confusionMatrixL
predL

library(caret)
library(class)
library(dplyr)
library(e1071)
library(FNN) 
library(gmodels) 
library(psych)

# processing data 
rta <- read.csv("RTA Dataset.csv")
rta_class <- rta[c("Age_band_of_driver", "Sex_of_driver", "Driving_experience", "Type_of_vehicle", "Defect_of_vehicle", "Accident_severity")]

# clean up unnecessary/unused data points
rta_class[rta_class == ""] <- NA
unique(rta_class$Age_band_of_driver)
rta_class[rta_class =="Unknown"] <- NA
unique(rta_class$Sex_of_driver)
rta_class[rta_class == "unknown"] <- NA
unique(rta_class$Type_of_vehicle)
rta_class[rta_class == "Other"] <- NA
unique(rta_class$Defect_of_vehicle)
rta_class <- na.omit(rta_class)
table(rta$Type_of_vehicle)

# dummy variale & binary columns
rta_class$Sex_of_driver <- ifelse(rta_class$Sex_of_driver == "Male", 0, 1)
rta_class$Age_band_of_driver <- dummy.code(rta_class$Age_band_of_driver)
rta_class$Driving_experience <- dummy.code(rta_class$Driving_experience)
rta_class$Sex_of_driver <- dummy.code(rta_class$Sex_of_driver)
rta_class$Type_of_vehicle <- dummy.code(rta_class$Type_of_vehicle)
rta_class$Defect_of_vehicle <- dummy.code(rta_class$Defect_of_vehicle)

# splitting dataset
set.seed(100)
acc_sev <- rta_class %>% select(Accident_severity)
rta_class <- rta_class %>% select(-Accident_severity)
train_index <- sample(seq_len(nrow(rta_class)), size=floor(0.8*nrow(rta_class)))
rta_train <- rta_class[train_index, ]
rta_test <- rta_class[-train_index, ]
acc_sev_train <- acc_sev[train_index, ]
acc_sev_test <- acc_sev[-train_index, ]

#knn
acc_sev_knn_pred <- knn(train=rta_train, test=rta_test, cl=acc_sev_train, k = 10)
acc_sev_test <- data.frame(acc_sev_test)
compare <- data.frame(acc_sev_knn_pred, acc_sev_test)
names(compare) <- c("Predicted_Accident_Severity", "Observed_Accident_Severity")
CrossTable(x = compare$Observed_Accident_Severity, y = compare$Predicted_Accident_Severity, prop.chisq=FALSE, prop.c=FALSE, prop.r=FALSE, prop.t=FALSE)

#########################################################################################################################
# Predicting Cause of Accident
library("tibble")
library(dplyr)
library(tree)
library(tidyr)

rta <- as.tibble(rta)

rta2 <- rta[, c("Age_band_of_driver", "Sex_of_driver", "Educational_level", "Driving_experience", "Cause_of_accident") ]

rta2 <- rta2[complete.cases(rta2), ]

rta2$Cause_of_accident[rta2$Cause_of_accident == "Changing lane to the left"  | rta2$Cause_of_accident =="Changing lane to the right"] <- "Changing lane"
rta2$Cause_of_accident[rta2$Cause_of_accident !="Changing lane"] <- "Other"

#rta2 <- subset(rta2, Age_band_of_driver != "Unknown" & Sex_of_driver != "Unknown" & Educational_level != "Unknown" & Driving_experience != "Unknown" & Cause_of_accident != "Other")
rta2 <- subset(rta2, Age_band_of_driver != "Unknown" & Sex_of_driver != "Unknown" & Educational_level != "Unknown" & Driving_experience != "Unknown")

rta2$Cause_of_accident <- as.factor(rta2$Cause_of_accident)
rta2$Age_band_of_driver <- as.factor(rta2$Age_band_of_driver)
rta2$Educational_level <- as.factor(rta2$Educational_level)
rta2$Driving_experience <- as.factor(rta2$Driving_experience)
rta2$Sex_of_driver <- as.factor(rta2$Sex_of_driver)

library(rpart)
set.seed(1)
train_id <- sample(1:nrow(rta2), nrow(rta2) * 0.7)
test <- rta2[-train_id,]
train <- rta2[train_id,]
cause.test <- test$Cause_of_accident
tree.rta2 <- rpart(Cause_of_accident~., train, method="class", control=rpart.control(minsplit=10, cp=0))

plot(tree.rta2)
text(tree.rta2, pretty=0)

tree.pred <- predict(tree.rta2, test, type="class")
table(tree.pred, cause.test)
mean(cause.test == tree.pred)

summary(train$Cause_of_accident)

# oversampling
library(ROSE)
set.seed(5)
over <- ovun.sample(Cause_of_accident~., data = train, method = "over", N = 4938*2)$data
table(over$Cause_of_accident)
tree.rta2 <- rpart(Cause_of_accident~., over, method="class", control=rpart.control(minsplit=10, cp=0))
tree.pred <- predict(tree.rta2, test, type="class")
table(tree.pred, cause.test)
mean(cause.test == tree.pred)

printcp(tree.rta2)
plotcp(tree.rta2)
set.seed(5)
over <- ovun.sample(Cause_of_accident~., data = train, method = "over", N = 4938*2)$data
tree.rta2 <- rpart(Cause_of_accident~., over, method="class", control=rpart.control(minsplit=10, cp=0.00060753))
tree.pred <- predict(tree.rta2, test, type="class")
table(tree.pred, cause.test)
mean(cause.test == tree.pred)

# undersampling
set.seed(5)
over <- ovun.sample(Cause_of_accident~., data = train, method = "under", N = 1812*2)$data
table(over$Cause_of_accident)
tree.rta2 <- rpart(Cause_of_accident~., over, method="class", control=rpart.control(minsplit=10, cp=0))
tree.pred <- predict(tree.rta2, test, type="class")
table(tree.pred, cause.test)
mean(cause.test == tree.pred)

printcp(tree.rta2)
plotcp(tree.rta2)
set.seed(10)
over <- ovun.sample(Cause_of_accident~., data = train, method = "under", N = 1812*2)$data
tree.rta2 <- rpart(Cause_of_accident~., over, method="class", control=rpart.control(minsplit=10, cp=0.00110375))
tree.pred <- predict(tree.rta2, test, type="class")
table(tree.pred, cause.test)
mean(cause.test == tree.pred)

# random forests
library(randomForest)
set.seed(1)
bag.mse = rep(0,4)
for(i in 1:4) {
  bag1 <- randomForest(Cause_of_accident~., data=over, mtry=i, importance=TRUE)
  bag1.pred <- predict(bag1, test, type="class")
  bag.mse[i] = mean(bag1.pred ==test$Cause_of_accident)
}
which.max(bag.mse)

bag <- randomForest(Cause_of_accident~., data=over, mtry=2, importance=TRUE)
bag.pred <- predict(bag, test)
table(bag.pred, cause.test)
mean(cause.test == bag.pred)
varImpPlot(bag)

