library(caret)
library(dplyr)
library(caTools)
library(gbm)
library(randomForest)
source("0_utils_functions.r")
# Load the three models
load("models/fit_gbm.RData")
load("models/fit_rf.RData")
load("models/fit_log_reg_boost.RData")
# Change the following line with the csv of your data.
all_data_tb <- read.csv(r"(CPOD_data.csv)", sep = ";")
# Processing and inference
all_data_tb_p <- preprocess_data(all_data_tb)
# change the argument "return_probs" to false if you want the raw probabilities
# instead of the predicted category.
predictions_df <- run_predictions(
  data = all_data_tb_p,
  threshold_file = "thresholds.csv",
  return_probs = "FALSE"
)
# We can do some post processing on the predictions dataframe.
# by removing the pal detections that occur alone ina single hour.
data_and_predictions <- bind_cols(all_data_tb, predictions_df) # combine the original data with predictions

data_and_predictions <- post_processing(data_and_predictions)

corr_comp <- with(
  data_and_predictions,
  cbind(table(ensamble), table(corrected_category))
)
colnames(corr_comp) <- c("Before Correction", "After Correction")
rownames(corr_comp) <- c("PAL", "porp")
corr_comp
