---
title: "Prediction Assignment"
author: "Vishal Sharma"
date: "Wednesday, December 17, 2014"
output: html_document
----------------------------

Build a machine learning algorithm to predict activity quality from activity monitors
-------------------------------------------------------------------------------------

**Add the required Libraries**


```r
library(Hmisc)
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
set.seed(2048)
options(warn=-1)
```

**STEP-1 :: Getting and Cleaning the data**

- 1- Read the Assignment data files from current working director
- 2- Some values contained a "#DIV/0!" , Those are replaced with NA
- 3- Casted all columns 8 to the end to be numeric
- 4- Chose a feature set that only included complete columns
- 5- Remove User name, timestamps and windows



```r
#Load the PML Training data set file
dataTraining <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
#Load the PML Test data set file
dataEval  <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
# Make Columns to be numeric from 8rt coloum onwards for PML Training data set file
for(i in c(8:ncol(dataTraining)-1)) {dataTraining[,i] = as.numeric(as.character(dataTraining[,i]))}
# Make Columns to be numeric from 8rt coloum onwards for PML Test data set file
for(i in c(8:ncol(dataEval)-1)) {dataEval[,i] = as.numeric(as.character(dataEval[,i]))}
#Chose a feature set that only included complete columns for PML Training data set file
feature_set <- colnames(dataTraining[colSums(is.na(dataTraining)) == 0])[-(1:7)]
model_data <- dataTraining[feature_set]
feature_set
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

```r
dim(dataTraining); dim(dataEval)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

**STEP-2 :: Partioning  the training set**
- Partioning Training data set into two data sets, 75% for training, 25% for testing


```r
inTrain  <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[inTrain ,]
testing <- model_data[-inTrain ,]
dim(training); dim(testing)
```

```
## [1] 14718    53
```

```
## [1] 4904   53
```
**STEP-3 :: Using ML algorithms for prediction:: Random Forests**

- 1- Use of parallel processing to build this model using registerDoParallel
- 2- Build 5 random forests with 150 trees each


```r
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe
rfModel <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {randomForest(x, y, ntree=ntree)                                                                                                        }
```

**STEP-4:: Predicting in-sample error in both Training and Test data**

```r
#Create Confusion Matrix for Training Data
predictionsTraining <- predict(rfModel, newdata=training)
confusionMatrix(predictionsTraining,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
#Create Confusion Matrix for Test Data
predictionsTesting <- predict(rfModel, newdata=testing)
confusionMatrix(predictionsTesting,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  942    6    0    0
##          C    0    3  848    8    2
##          D    0    0    1  796    2
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9922, 0.9965)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9918   0.9900   0.9956
## Specificity            0.9989   0.9985   0.9968   0.9993   1.0000
## Pos Pred Value         0.9971   0.9937   0.9849   0.9962   1.0000
## Neg Pred Value         1.0000   0.9982   0.9983   0.9981   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1729   0.1623   0.1829
## Detection Prevalence   0.2853   0.1933   0.1756   0.1629   0.1829
## Balanced Accuracy      0.9994   0.9956   0.9943   0.9947   0.9978
```
**Note On :: how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did**

The two machine learning algorithm is used for the cross validation : regression tree (rpart) and random forest.
Regressioin Tree is not mentioned to keep document short. but it resulted in 89% accuracy. So 
the random forest algorithm is picked up.
The Machine Learning algorithm as Random forest resulted in the 99.5% accuracy on the test data as seen in the result output of confusion matrix mentioned above. So it is assumed to be worked very well on test data set given for the assignment. After running the below code and submission it proved also.


**Generating Files to submit as answers for the Assignment**

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
x <- dataEval
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rfModel, newdata=x)

pml_write_files(answers)
```
Credit Note : I am thankfull to all the people for the posts they posted ,  i looked to understand the concepts and make me capable to complete the assignment.

