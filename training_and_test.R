library(caret)
library(glue)
library(plotROC)
library(pROC)
library(ggplot2)
# Load the example clicks dataset
data <- read.csv("examples_clicks.csv")
# data <- read.csv("your_labelled_data.csv")
# one hot encode the TrClass variable
data$TrClassH <- ifelse(data$TrClass == "High", 1, 0)
data$TrClassM <- ifelse(data$TrClass == "Mod", 1, 0)
# select only the features necessary for training
data <- data[, c(
     "station", "Start", "TrDur_us",
     "NofClx", "nActualClx", "Clx.s", "ICIgood",
     "modalKHz", "avSPL", "MaxICI_us", "MinICI_us",
     "MMM", "TimeLost", "ClksThisMin", "LastICI_us",
     "ICI_rising", "avEndF", "MinF", "MaxF",
     "avNcyc", "MaxNcyc", "MaxSPL", "NofClstrs",
     "avClstrN.10", "Frange", "avSlope1",
     "avSlope2", "avBW", "TrClassH", "TrClassM", "click"
)]
# split the data into train and test
train_index <- createDataPartition(data$click, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
station_train <- train_data$station
station_test <- test_data$station
# remove the station column from both train and test
train_data <- train_data[, -1]
test_data <- test_data[, -1]
# make the response a factor
train_data$click <- factor(train_data$click, levels = c("pal", "porp"))
test_data$click <- factor(test_data$click, levels = c("pal", "porp"))
# define the weights
w_porp <- length(train_data$click) / table(train_data$click)["porp"]
w_pal <- length(train_data$click) / table(train_data$click)["pal"]
weights <- ifelse(train_data$click == "pal", w_pal, w_porp)
# define the fit control parameters, 10 fold cross validation
# repeated 10 times, returning class probabilites
fit_control <- trainControl(
     method = "repeatedcv",
     number = 10,
     repeats = 10,
     ## Estimate class probabilities
     classProbs = TRUE,
     ## Evaluate performance using
     ## the following function
     summaryFunction = twoClassSummary,
     savePredictions = TRUE
)

# Hyperparemeter grids for performing grid search for the  three models

tunegrid_gbm <- expand.grid(
     interaction.depth = c(1, 5, 9),
     n.trees = (1:30) * 50,
     shrinkage = 0.1,
     n.minobsinnode = 20
)
tunegrid_rf <- data.frame(.mtry = (1:28))
tunegrid_lgr <- data.frame(.nIter = (1:100))

set.seed(825)
fit_gbm <- train(click ~ .,
     data = train_data,
     method = "gbm",
     trControl = fit_control,
     verbose = TRUE,
     tuneGrid = tunegrid_gbm,
     ## Specify which metric to optimize
     metric = "ROC",
     weights = weights
)
fit_rf <- train(click ~ .,
     data = train_data,
     method = "rf",
     metric = "ROC",
     verbose = TRUE,
     trControl = fit_control,
     tuneGrid = tunegrid_rf,
     weights = weights
)
beepr::beep()
fit_log_reg_boost <- train(click ~ .,
     data = train_data,
     method = "LogitBoost",
     metric = "ROC",
     verbose = TRUE,
     trControl = fit_control,
     tuneGrid = tunegrid_lgr,
     weights = weights
)

confusionMatrix(
     data = fit_log_reg_boost$pred$pred,
     reference = fit_log_reg_boost$pred$obs
)
confusionMatrix(
     data = predict(
          fit_log_reg_boost,
          subset(test_data, select = -click)
     ),
     reference = test_data$click
)

save(fit_log_reg_boost, file = "log_reg_boosted_gridsearch2.RData")

# the best tune for each model is explained here
sapply(list(gbmfit, fit_rf, fit_log_reg_boost), \(x) x$bestTune)




# build roc curves for all the classifiers

# roc_gbm
indices_gbm <- dplyr::filter(
     gbmfit$pred,
     n.trees == 500,
     interaction.depth == 1,
     shrinkage == 0.1,
     n.minobsinnode == 50
)

indices_rf <- dplyr::filter(fit_rf$pred, mtry == 9)
indices_lgr <- dplyr::filter(fit_log_reg_boost$pred, nIter == 31)

gbm_curve <- roc(indices_gbm$obs, indices_gbm$porp)
rf_curve <- roc(indices_rf$obs, indices_rf$porp)
lgr_curve <- roc(indices_lgr$obs, indices_lgr$porp)

windows()
par(mfrow = c(1, 3))
plot(gbm_curve,
     print.auc = TRUE, auc.polygon = TRUE,
     auc.polygon.alpha = 0.2, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "lightgreen", print.thres = TRUE, main = "GBM"
)

plot(rf_curve,
     print.auc = TRUE, auc.polygon = TRUE,
     auc.polygon.alpha = 0.2, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "lightblue", print.thres = TRUE, main = "RF"
)

plot(lgr_curve,
     print.auc = TRUE, auc.polygon = TRUE,
     auc.polygon.alpha = 0.2, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "lightpink", print.thres = TRUE,
     main = "Boosted Logistic Regression"
)

# fin the best threshold for the gbm
th_gbm <- coords(gbm_curve, "best",
     ret = c("threshold", "sensitivity", "specificity"),
     best.method = "youden"
)

# fin the best threshold for the rf
th_rf <- coords(rf_curve, "best",
     ret = c("threshold", "sensitivity", "specificity"),
     best.method = "youden"
)


# fin the best threshold for the logreg
th_lgr <- coords(lgr_curve, "best",
     ret = c("threshold", "sensitivity", "specificity"),
     best.method = "youden"
)

write.csv(c(
     "gbm" = th_gbm[1],
     "rf" = th_rf[1],
     "lgr" = th_lgr[1]
), "thresholds2.csv")
