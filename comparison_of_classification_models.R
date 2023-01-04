# Comparison of Classification Models. 

# The objective is to compare the following classification algorithms on a particular dataset:
# 1.	Linear Discriminant Analysis
# 2.	Quadratic Discriminant Analysis
# 3.	K Nearest Neighbors
# 4.	Random Forest


library(caret)

# We will use the caret library for this project as it's syntax is simple
# and straightforward.

# Loading the data from the .txt file.
project2 = read.csv('project2.txt', header = FALSE, sep = ",")

# Converting the response as factors so that classification models can be fit. 
project2$V5 = as.factor(project2$V5)

# Viewing the table
View(project2)

# Viewing the dimensions of the table 
dim(project2)

# Viewing the column names.
names(project2)

# Viewing the summary of the dataset. 
summary(project2)

# Plotting the variables 
pairs(project2)

# Viewing the correlation matrix
cor(project2[-5])


#### Linear Discriminant Analysis
# First we fit the LDA model 

# Splitting train and test data. 
set.seed(12)
train.index <- createDataPartition(project2[,"V5"],p=0.8,list=FALSE)
project2.trn <- project2[train.index,]
project2.tst <- project2[-train.index,]

ctrl  <- trainControl(method  = "cv",number  = 10) 

# Fitting the model
fit.cv <- train(V5 ~ ., data=project2.trn, method="lda",
                trControl = ctrl)

# Making predictions on the test data
pred <- predict(fit.cv,project2.tst)

# Viewing the cross validation results. 
confusionMatrix(table(project2.tst[,"V5"],pred))

# Cross-Validated Accuracy for LDA is **** 0.9635 ****
# Observing the Confusion matrix we can tell that there are **** 10 false negatives ****
# out of a total of 274 predictions
# Here we assume 0 as False and 1 as True
print(fit.cv)


#### Quadratic Discriminant Analysis
# Now we fit the QDA model 

# Splitting train and test data. 
set.seed(12)
train.index <- createDataPartition(project2[,"V5"],p=0.8,list=FALSE)
project2.trn <- project2[train.index,]
project2.tst <- project2[-train.index,]

ctrl  <- trainControl(method  = "cv",number  = 10) 

# Fitting the model
fit.cv <- train(V5 ~ ., data=project2.trn, method="qda",
                trControl = ctrl)

# Making predictions on the test data
pred <- predict(fit.cv,project2.tst)

# Viewing the cross validation results.
confusionMatrix(table(project2.tst[,"V5"],pred))

# Cross-Validated Accuracy for QDA is **** 0.9708 ****
# Observing the Confusion matrix we can tell that there are **** 8 false negatives ****
# out of a total of 274 predictions
# Here we assume 0 as False and 1 as True
print(fit.cv)



#### K Nearest Neighbors
# Next we fit the KNN model 

# Splitting train and test data.
set.seed(12)
train.index <- createDataPartition(project2[,"V5"],p=0.8,list=FALSE)
project2.trn <- project2[train.index,]
project2.tst <- project2[-train.index,]

ctrl  <- trainControl(method  = "cv",number  = 10) 

# Fitting the model using 10 fold CV as defined above.
fit.cv <- train(V5 ~ ., data = project2.trn, method = "knn",
  trControl = ctrl, 
  preProcess = c("center","scale"), 
  tuneGrid =data.frame(k=seq(1,35,by=1)))

# As can be seen from the plot later, the maximum accuracy is achieved at k = 5

# Making predictions on the test data
pred <- predict(fit.cv,project2.tst)

# Viewing the cross validation results.
confusionMatrix(table(project2.tst[,"V5"],pred))

# Cross-Validated Accuracy for KNN is **** 0.9964  ****
# Observing the Confusion matrix we can tell that there are **** 1 false negatives ****
# out of a total of 274 predictions
# Here we assume 0 as False and 1 as True

print(fit.cv)
# The maximum accuracy is achieved at k = 5
plot(fit.cv)
# The same is apparent from the plot.


#### Random Forest

# Next we fit the Random Forest model

# Splitting train and test data.
set.seed(0)
train.index <- createDataPartition(project2[,"V5"],p=0.8,list=FALSE)
project2.trn <- project2[train.index,]
project2.tst <- project2[-train.index,]

ctrl  <- trainControl(method  = "cv",number  = 10) 

# Fitting the model using 10-fold CV as defined above. 
fit.cv <- train(V5 ~ ., data = project2.trn, method = "rf",
                trControl = ctrl, 
                tuneLength = 50)

# 3 parameters were selected 

print(fit.cv)

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 2.

plot(fit.cv)
# Same is apparent from the plot. 

fit.cv$results
# Accuracy is maximised with 3 variables. 

# Making predictions on the test data
pred <- predict(fit.cv,project2.tst)

# Viewing the cross validation results.
confusionMatrix(table(project2.tst[,"V5"],pred))

# Cross-Validated Accuracy for KNN is **** 0.9891  ****
# Observing the Confusion matrix we can tell that there are **** 3 false negatives ****
# out of a total of 274 predictions
# Here we assume 0 as False and 1 as True

# Let us now check the variable importance.
# As can be seen from the plot and the table, V1 is the most important variable in the dataset. 
print(varImp(fit.cv)) # Variable importance
plot(varImp(fit.cv))

#### Conclusion ####

# In conclusion, KNN appears to produce the best results with 99.64% Accuracy.









