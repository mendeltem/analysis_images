setwd("~/Documents/work/mendel_exchange")
#claudius
#install.packages("glmnet", repos = "http://cran.us.r-project.org")

rm(list=ls())
#install.packages('e1071', dependencies=TRUE)
install.packages('caretEnsemble')
library(dplyr)
library(tidyverse)
library(ggplot2)
library(FactoMineR)
library(glmnet)
library(vtreat)
library(mlbench)
library(caret)
library(caretEnsemble)
set.seed(42)
library("factoextra")
library(mgcv)
library(broom)

#die Daten laden
filelist <- list.files(path = "saved/color_high-pass/csv/",pattern = "Output", full.names=T)

df = data.frame()

for (k in 1:length(filelist)){
  temp <- read.csv(filelist[k])
  df = bind_rows(df,temp)
  
  print(k)
  if (k == 30){
    break
  }
}
#filter the specific experiment type
experiment_t1 = df %>% filter( (colorimages == 1) 
                         & (maskregion == 0) 
                         & (masktype == 0))

dim(experiment_t1)

#get only the features from the last layer
part11 = experiment_t1[,c(19,(10081-2047):10081)]
part11 = experiment_t1[,c(19,(10081-2047):10081)]
#from first layer
part1 = experiment_t1[,c(19,(34):(34+255))]
part2 = experiment_t1[,c(19,(34+256):(34+255+255))]

colnames(part2)
#get vector fixdur
#y_11 = experiment_t1[,19]

inTrain = round(nrow(part11[19]) * 2/3)

training <- part11[1:inTrain,]
test <- part11[inTrain:nrow(part11),]
trainMDRR <- part11[1:inTrain,19]
testMDRR <- part11[inTrain:nrow(part11),19]




#prepare data for layer 1
inTrain_1 = round(nrow(part1) * 0.9)
training_1 <- part1[1:inTrain_1,]
test_1 <- part1[inTrain_1:nrow(part1),]

#prepare data for layer 2
inTrain_2 = round(nrow(part2) * 0.9)
training_2 <- part2[1:inTrain_2,]
test_2 <- part2[inTrain_2:nrow(part2),]


#prepare data for layer 2
inTrain_11 = round(nrow(part11) * 0.9)
training_11 <- part11[1:inTrain_11,]
test_11 <- part2[inTrain_11:nrow(part11),]


dim(training_2)
colnames(training_2)



# Create variables
i <- y ~ .
j <- y ~ . + .
k <- y ~ . + . + .

# Concatentate
formulae <- list(as.formula(i),as.formula(j),as.formula(k))

# Double check the class of the list elements
class(formulae[[1]])


#witch zva deleting variables nearly constant
model_corr_pre_log_glmnet = train(log(fixdur) ~ ., training_11, 
               method="glmnet",
               trControl= trainControl(method = "cv",
                                       number = 3, 
                                       verboseIter = TRUE),
               preProcess = c("zv","medianImpute", "center", "scale","pca")
)

model_corr_pre$results


summary(model_corr_pre)

summary(model_corr_pre)
summary(model_corr_pre_log_glmnet)


max(model_corr_pre_log_glmnet$results$Rsquared)


#witch zva deleting variables nearly constant
model_lmnet = train(fixdur ~ ., training_11, 
                   method="glmnet"
)

model_lmnet

summary(model_lmnet)


dim(training_11)
#witch zva deleting variables nearly constant
model_2= train(fixdur ~ ., training_2, 
               method="logicBag",
               preProcess = c("zv","medianImpute", "center", "scale")
)

ymodel_2


simple_lm= lm(fixdur ~ ., training_2)



model3 = train(fixdur ~ log,
               training_2, 
                   method="logicBag",
                   trControl= trainControl(method = "cv",
                                           number = 1, 
                                           verboseIter = TRUE)
                   )

str(training_2[1])


summary(simple_lm)

summary(model_corr)

model_11

#Ridge or Lasso
# Create custom trainControl: myControl

# Fit glmnet model: model
model2 =train(fixdur ~ ., training, 
              method="glmnet",
              trControl =trainControl(method="cv", number = 10, verboseIter = T))



#witch zva deleting variables nearly constant
model4 = train(fixdur ~ ., training, 
               method="lm",
               trControl= trainControl(method = "cv",
                                       number = 10, 
                                       repeats = 5,
                                       verboseIter = TRUE),
               preProcess = c("nzv","medianImpute", "center", "scale","pca")
)

#witch zva deleting variables nearly constant
model6 = train(fixdur ~ ., training, 
               method="gam",
               tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1 , length = 100) ),
               trControl= trainControl(method = "cv",
                                       number = 10, 
                                       repeats = 10,
                                       verboseIter = TRUE),
               preProcess = c("zv","medianImpute", "center", "scale","pca")
)
#witch zva deleting variables nearly constant
model_glmnet_11 = train(fixdur ~ ., training, 
                        method="glmnet",
                        tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1 , length = 100) ),
                        trControl= trainControl(method = "cv",
                                                number = 10, 
                                                repeats = 10,
                                                verboseIter = TRUE),
                        preProcess = c("zv","medianImpute", "center", "scale","pca")
)

#witch zva deleting variables nearly constant
model_glmnet_1 = train(fixdur ~ ., training_1, 
               method="glmnet",
               tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1 , length = 100) ),
               trControl= trainControl(method = "cv",
                                       number = 10, 
                                       repeats = 10,
                                       verboseIter = TRUE),
               preProcess = c("zv","medianImpute", "center", "scale","pca")
)
#witch zva deleting variables nearly constant
model_glmnet_2 = train(fixdur ~ ., training_2, 
                       method="glmnet",
                       tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1 , length = 100) ),
                       trControl= trainControl(method = "cv",
                                               number = 10, 
                                               repeats = 10,
                                               verboseIter = TRUE),
                       preProcess = c("zv","medianImpute", "center", "scale","pca")
)
#witch zva deleting variables nearly constant
model_gam_2 = train(fixdur ~ ., training_2[,1:40], 
                       method="gam",
                       trControl= trainControl(method = "cv",
                                               number = 2, 
                                               verboseIter = TRUE),
                       preProcess = c("zv","medianImpute", "center", "scale")
)
summary(model_gam_2)
dim(training_2)
#layer 2
gam1 = gam(fixdur ~ s(X0.1)+s(X1.1)+s(X2.1), data = training_2)
gam2 = gam(fixdur ~ X0.1 + X1.1+X2.1, data = training_2)

summary(gam1)
summary(gam2)

anova(gam1,gam2)


print(model_gam_2)
summary(model_gam_2)

dim(training_1)

print(test_class_cv_model)

#Modelle Vergleichen
model_list = list(  model_glmnet_11 = model5,
                    model_glmnet_1   =model_glmnet_1,
                    model_glmnet_2  =model_glmnet_2
                    )

resamps = resamples(model_list)


summary(resamps)



#-----------------------------------------------------------------------
dim(part11)
#deleting near zero variance variables
nzv <- nearZeroVar(part11)
filtered_part11 <- part11[, -nzv]
dim(filtered_part11)

# calculate correlation matrix
corrMatrix_part11 = cor(filtered_part11)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated_part11 <- findCorrelation(corrMatrix_part11, cutoff=0.75)

print(highlyCorrelated_part11)

filtered2_part11 <- filtered_part11[,-highlyCorrelated_part11]

dim(filtered2_part11)



# calculate the pre-process parameters from the dataset
pre_part11 <- preProcess(filtered2_part11, method=c("scale","center"))



# transform the dataset using the parameters
transformed_part11 <- predict(pre_part11, filtered2_part11)

dim(transformed_part11)

#preprocess to set training and test data
colnames(experiment_t1[19])


inTrain = round(nrow(experiment_t1[19]) * 2/3)


training <- transformed_part11[1:inTrain,]
test <- transformed_part11[inTrain:nrow(experiment_t1[19]),]
trainMDRR <- experiment_t1[1:inTrain,19]
testMDRR <- experiment_t1[inTrain:nrow(experiment_t1[19]),19]



preProcValues <- preProcess(training, method = c("center", "scale"))

trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, test)










#delete na
#df_f = df %>% drop_na()
df_f <-na.omit(control)
str(control)
df_2 = PCA(control[,34:10081],scale.unit=TRUE, ncp=1000, graph = F)



# Get the variance of the first 3 new dimensions.
df_2$eig[,2][1000]
  df_2$eig[,2][1:100]
# Get the cumulative variance.
df_2$eig[,3][936]
df_2$eig[,3][1:936]

#summary(df_2, nbelements = 10)
#fviz_screeplot(df_2, ncp=5)
#Regulization

dim(x.train)
dim(df)

str(df[1:2000,34:10081])

matrix = data.matrix(df[1:2000,34:10081])

str(df[1:2000,19])

#Train Test Data
x.train <- data.matrix(control[1:2000,34:10081])
x.test <- data.matrix(control[2001:2865,34:10081])
y.train <- control[1:2000,19]
y.test <- control[2001:2865,19]

dim(x.train)
dim(x.test)

str(y.train)
str(y.test)

#lass or ridge regression
alpha1.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
                        alpha=1, family="gaussian")

summary(alpha1.fit)

alpha1.fit$lambda


alpha1.fit$cvm
alpha1.fit$lambda

alpha1.predicted <- 
  predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)

mean((y.test - alpha0.predicted)^2)


list.of.fits <- list()
for (i in 0:10) {
  ## Here's what's going on in this loop...
  ## We are testing alpha = i/10. This means we are testing
  ## alpha = 0/10 = 0 on the first iteration, alpha = 1/10 = 0.1 on
  ## the second iteration etc.
  
  ## First, make a variable name that we can use later to refer
  ## to the model optimized for a specific alpha.
  ## For example, when alpha = 0, we will be able to refer to 
  ## that model with the variable name "alpha0".
  fit.name <- paste0("alpha", i/10)
  
  ## Now fit a model (i.e. optimize lambda) and store it in a list that 
  ## uses the variable name we just created as the reference.
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure="mse", alpha=i/10, 
              family="gaussian")
}
## Now we see which alpha (0, 0.1, ... , 0.9, 1) does the best job
## predicting the values in the Testing dataset.
results <- data.frame()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  
  ## Use each model to predict 'y' given the Testing dataset
  predicted <- 
    predict(list.of.fits[[fit.name]], 
            s=list.of.fits[[fit.name]]$lambda.1se, newx=x.test)
  
  ## Calculate the Mean Squared Error...
  mse <- mean((y.test - predicted)^2)
  
  ## Store the results
  temp <- data.frame(alpha=i/10, mse=mse, fit.name=fit.name)
  results <- rbind(results, temp)
}
results


res = cv.glmnet(x.train, y.train, type.measure="mse", alpha=1, 
          family="gaussian")

summary(res)


dim(x.train)


simple_gam_model1 = gam(fixdur ~ s(X0.1)+ s(X1.1) + ti(X0.1,X1.1), data = training_2, method ="REML")

summary(simple_gam_model1)




sm = gam(fixdur ~ X0.1, data= training_2)

plot(training_2$fixdur, training_2$X0.1) + abline(coef(sm), col = "red")
summary(sm)



concurvity(simple_gam_model1)

gam.check(simple_gam_model1)


plot(simple_gam_model1, page=1 , scheme=2)

plot(simple_gam_model1, residuals = TRUE, pch = 1, cex = 1)


plot(simple_gam_model1,pages = 1 ,all.terms = TRUE,shade=TRUE ,shade.col = "hotpink")


predicted = predict(simple_gam_model1, data = test_2)


res = training_2$fixdur - predicted


ss_res = sum(res ^ 2)
ss_tot = sum( ( training_2$fixdur - mean(training_2$fixdur) ^2 ))

r_sqr = 1 - ss_res / ss_tot

print(r_sqr)

rsme = sqrt( mean ( res ^2 ))


  # 
# 
# #cross validation
# 
# # Load the package vtreat
# library(vtreat)
# 
# # mpg is in the workspace
# summary(mpg)
# 
# # Get the number of rows in mpg
# nRows <- nrow(mpg)
# 
# # Implement the 3-fold cross-fold plan with vtreat
# splitPlan <- kWayCrossValidation(nRows,3, NULL, NULL )
# 
# # Examine the split plan
# str(splitPlan)
# 
# 
# # mpg is in the workspace
# summary(mpg)
# 
# # splitPlan is in the workspace
# str(splitPlan)
# 
# # Run the 3-fold cross validation plan from splitPlan
# k <- 3 # Number of folds
# mpg$pred.cv <- 0 
# for(i in 1:k) {
#   split <- splitPlan[[i]]
#   model <- lm(cty~hwy, data = mpg[split$train, ])
#   mpg$pred.cv[split$app] <- predict(model, newdata = mpg[split$app,])
# }
# 
# # Predict from a full model
# mpg$pred <- predict(lm(cty ~ hwy, data = mpg))
# 
# # Get the rmse of the full model's predictions
# rmse(mpg$pred, mpg$cty)
# 
# # Get the rmse of the cross-validation predictions
# rmse( mpg$pred.cv, mpg$cty)
