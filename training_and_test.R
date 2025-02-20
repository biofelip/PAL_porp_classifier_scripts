library(caret)
library(glue)
library(plotROC)
library(pROC)
library(ggplot2)
library(cowplot)
source("utils_functions.r")
# Load the example clicks dataset
train_data <- read.csv("training_set.csv")
test_data <- read.csv("Test_Dataset_complete.csv")
# one hot encode the TrClass variable
train_data$TrClassH <- ifelse(train_data$TrClass == "High", 1, 0)
train_data$TrClassM <- ifelse(train_data$TrClass == "Mod", 1, 0)
# select only the features necessary for training
train_data <- train_data[, c(
     "Start", "TrDur_us",
     "NofClx", "nActualClx", "Clx.s", "ICIgood",
     "modalKHz", "avSPL", "MaxICI_us", "MinICI_us",
     "MMM", "TimeLost", "ClksThisMin", "LastICI_us",
     "ICI_rising", "avEndF", "MinF", "MaxF",
     "avNcyc", "MaxNcyc", "MaxSPL", "NofClstrs",
     "avClstrN.10", "Frange", "avSlope1",
     "avSlope2", "avBW", "TrClassH", "TrClassM", "click"
)]

test_data <- test_data[, -1] # remove the index from test set
# make the click a factor with the second level being the porp so its the positive class
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
     interaction.depth = c(2, 5, 9),
     n.trees = seq(100, 1000, by = 50),
     shrinkage = seq(0.1:0.01, by = 0.2),
     n.minobsinnode = seq(10, 60, by = 5)
)
tunegrid_rf <- data.frame(.mtry = (1:28))
tunegrid_lgr <- data.frame(.nIter = (1:100))
set.seed(825) # seed for reproducibility
fit_gbm <- train(click ~ .,
     data = train_data,
     method = "gbm",
     trControl = fit_control,
     tuneGrid = tunegrid_gbm,
     ## Specify which metric to optimize
     metric = "ROC",
     weights = weights
)
save(fit_gbm, file = "fit_gbm.RData")
fit_rf <- train(click ~ .,
     data = train_data,
     method = "rf",
     metric = "ROC",
     verbose = TRUE,
     trControl = fit_control,
     tuneGrid = tunegrid_rf,
     weights = weights
)
save(fit_rf, file = "fit_rf.RData")
fit_log_reg_boost <- train(click ~ .,
     data = train_data,
     method = "LogitBoost",
     metric = "ROC",
     verbose = TRUE,
     trControl = fit_control,
     tuneGrid = tunegrid_lgr,
     weights = weights
)
save(fit_log_reg_boost, file = "fit_log_reg_boost.RData")
# the best tune for each model is explained here

sapply(list(fit_gbm, fit_rf, fit_log_reg_boost), \(x) x$bestTune)

### Plotting results
# Confusion matrices for Validation
ensamble <- list(fit_gbm, fit_rf, fit_log_reg_boost)
cf_mat_val <- lapply(ensamble, confusionMatrix.train, norm = "none")
cf_plots <- lapply(cf_mat_val, plot_cf)
plot_grid(plotlist = cf_plots, nrow = 3, labels = c("GBM", "RF", "LBR"))
# Confusion matrices for the test set
cf_mat_t <- lapply(ensamble, confusionMatrix, newdata = test_set, norm = "none")
cf_plots_t <- lapply(cf_mat_t, plot_cf)
plot_grid(plotlist = cf_plots_t, nrow = 3, labels = c("GBM", "RF", "LBR"))
# build roc curves for all the classifiers
# Best models
indices_gbm <- dplyr::filter(
     fit_gbm$pred,
     n.trees == fit_gbm$bestTune$n.trees,
     interaction.depth == fit_gbm$bestTune$interaction.depth,
     shrinkage == fit_gbm$bestTune$shrinkage,
     n.minobsinnode == fit_gbm$bestTune$n.minobsinnode
)
indices_rf <- dplyr::filter(fit_rf$pred, mtry == fit_rf$bestTune$mtry)
indices_lgr <- dplyr::filter(
     fit_log_reg_boost$pred,
     nIter == fit_log_reg_boost$bestTune$nIter
)
# validation curves
gbm_curve <- roc(indices_gbm$obs, indices_gbm$porp)
rf_curve <- roc(indices_rf$obs, indices_rf$porp)
lgr_curve <- roc(indices_lgr$obs, indices_lgr$porp)
# test set curves
gbm_curve_t <- roc(test_data$click, predict.train(fit_gbm, newdata = test_data, type = "prob")$porp)
rf_curve_t <- roc(test_data$click, predict.train(fit_rf, newdata = test_data, type = "prob")$porp)
lgr_curve_t <- roc(test_data$click, predict.train(fit_log_reg_boost, newdata = test_data, type = "prob")$porp)
# helper function
plot_curve <- function(roc_c, main, color) {
     plot(roc_c,
          print.auc = TRUE, auc.polygon = TRUE,
          auc.polygon.alpha = 0.2, grid = c(0.1, 0.2),
          grid.col = c("green", "red"), max.auc.polygon = TRUE,
          auc.polygon.co = color, print.thres = TRUE, main = main
     )
}
windows()
par(mfrow = c(3, 2))
plot_curve(gbm_curve, "GBM (Validation)", color = "lightgreen")
plot_curve(gbm_curve_t, "GBM (Test)", color = "lightblue")
plot_curve(rf_curve, "RF (Val)", color = "lightgreen")
plot_curve(rf_curve_t, "RF (Test)", color = "lightblue")
plot_curve(lgr_curve, "LGR (Val)", color = "lightgreen")
plot_curve(lgr_curve_t, "LGR (Test)", color = "lightblue")

# find the best threshold for the models
bm_curves <- list(gbm_curve, rf_curve, lgr_curve)

trhs_all <- lapply(bm_curves, \(x) {
     coords(x, "best",
          ret = c("threshold", "sensitivity", "specificity"),
          best.method = "youden"
     )
})

write.csv(c(
     "gbm" = trhs_all[[1]][1],
     "rf" = trhs_all[[2]][1],
     "lgr" = trhs_all[[3]][1]
), "thresholds.csv", )
