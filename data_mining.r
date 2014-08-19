# This assignment takes ocean sensor data and runs through
# several steps, to build decision trees and random forests type algorithims 
# to see what machine learning techniques work best
# The repository for this assignment is at https://github.com/uwescience/datasci_course_materials
# under assignment 5


# load data
#setwd("C:/coursera/data_science/datasci_course_materials/assignment5")
setwd("/home/edecambra/datasci_course_materials/assignment5")
system("ls")
data <- read.csv("seaflow_21min.csv")
summary(data) #note that pop = synecho has 18,146 obs fsc_small qt3 is 39,184

# split data into test set
sample_int <- nrow(data)/2 - .5  #sub .5 to keep as integers
set.seed(1)
test_index <- sample(row.names(data), size = sample_int)
test_index <- sort(as.integer(test_index))
test_set <- data[test_index,]
train_set <- data[!(row.names(data) %in% test_index),]
summary(train_set); summary(test_set)

# plot the chl_small versus pe and color by pop
require(ggplot2)
plot(chl_small ~ pe, data = data)
ggplot(data, aes(chl_small, pe )) + # this plots the scatter pe on y and chl_small on x
  geom_point(aes(colour = pop))  # this adds color by pop to the scatter

# Traning a decision tree
require(rpart)
form <- formula(pop ~ fsc_small + fsc_perp + chl_small + pe + 
                           chl_big + chl_small)
dtree <- rpart(form, method = "class", data = train_set)
print(dtree)

# Evaluating the previous tree
eval_trees <- predict(dtree, newdata = test_set, type = "class") #note the use of "newdata"
print(eval_trees[1:10])
test_boo1 <- test_set$pop == eval_trees
error_rate1 <- sum(test_boo1)/nrow(test_set)
print(error_rate1) # prints an error rate of .8520085, which is pretty good

# Random Forest Supervised Learning model
require(randomForest)
rforest <- randomForest(form, data = train_set)
eval_forest <- predict(rforest, newdata = test_set, type = "class")
test_boo2 <- test_set$pop == eval_forest
error_rate2 <- sum(test_boo2)/nrow(test_set)
print(error_rate2) # prints an error rate of .9196041, much better than the trees
importance(rforest)

# Support Vector Machine algorithims
require(e1071)
supportVM <- svm(form, data = train_set)
eval_svm <- predict(supportVM, newdata = test_set, type = "class")
test_boo3 <- test_set$pop == eval_svm
error_rate3 <- sum(test_boo3)/nrow(test_set)
print(error_rate3) # prints an error rate .9212629

# create confusion matrices for each learning method
table(pred = eval_trees, true = test_set$pop)
table(pred = eval_forest, true = test_set$pop)
table(pred = eval_svm, true = test_set$pop)

# Checking for non continuous variable
hist(data$fsc_small)
hist(data$fsc_perp)
hist(data$fsc_big) # this variable takes on a few discrete values
hist(data$pe)
hist(data$chl_small)
hist(data$chl_big)

#plotting fsc_big versus time shows a band in error, corresponding to file_id == 208
plot(fsc_big ~ time, data = data)

# subsetting the main data to exclude this corrupt file, and repeating for VSC model+
# Note that this reclasifies all the previous variables in memory
clean_data <- data[ !data$file_id == 208, ]
sample_int <- nrow(clean_data)/2 
set.seed(1)
test_index <- sample(row.names(clean_data), size = sample_int)
test_index <- sort(as.integer(test_index))
test_set <- clean_data[test_index,]
train_set <- clean_data[!(row.names(clean_data) %in% test_index),]

require(e1071)
supportVM <- svm(form, data = train_set)
eval_svm <- predict(supportVM, newdata = test_set, type = "class")
test_boo3 <- test_set$pop == eval_svm
error_rate3 <- sum(test_boo3)/nrow(test_set)
print(error_rate3) # the new error rate is .918274
