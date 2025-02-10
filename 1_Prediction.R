# This script takes raw data from a folder and produces a dataframe with
# the raw data and the predicted class for each model and one ensmeble
# predicition.

library(caret)
library(dplyr)
library(caTools)
library(gbm)
library(randomForest)
source("utils_functions.r")

# load the three models

load("log_reg_boosted_gridsearch2.RData")
load("rf_gridsearch2.RData")
load("gbmfit2.RData")




# # region for the testing the historic data set of the baltic and north sea
# all_data_folder <- "ALLDATA\\north sea and baltic sea"
# # get all the files in the folder
# all_files <- list.files(all_data_folder, pattern = "*.csv", full.names = TRUE)
# all_data  <- lapply(all_files, read.csv, sep=";")
# all_data[[1]]$sea  <- "baltic"
# all_data[[2]]$sea  <- "north"
# # Join them together in a single dataframe
# all_data_tb  <- do.call(rbind, all_data)
# colnames(all_data_tb)
# seas <- all_data_tb$sea
# all_data_tb  <- preprocess_data(all_data_tb)
# predictions_df  <- run_predictions(all_data_tb, "thresholds2.csv")
# predictions_df$sea  <- seas
# with(predictions_df, table(sea, ensamble))
# write.csv(cbind(all_data_tb, predictions_df),
#           file = "prediction_historic_baltic_north.csv", row.names = FALSE)
# #endregion

# region fir runing the whole dataset and saving the result
# all_data_folder <- r"(C:\Users\247404\Documents\2024\jeff_porpoise_detector\ALLDATA)"
# # create a dir inside to save the result
# # dir.create(file.path(all_data_folder, "Predictions2"))
# # get all the files in the folder
# all_files <- list.files(all_data_folder, pattern = "*.txt", full.names = TRUE)
# all_data  <- lapply(all_files, read.delim)
# # Join them together in a single dataframe
# all_data_tb  <- do.call(rbind, all_data)
# clicks  <- all_data_tb$clicks
# # One hot encode the column TrClass in the dataframe
# all_data_tb$TrClassH  <- ifelse(all_data_tb$TrClass == "High", 1, 0)
# all_data_tb$TrClassM  <- ifelse(all_data_tb$TrClass == "Mod", 1, 0)
# # selec the correct columns
# all_data_tb <- all_data_tb[, c("Start", "TrDur_us",
#                                "NofClx", "nActualClx", "Clx.s", "ICIgood",
#                                "modalKHz", "avSPL", "MaxICI_us", "MinICI_us",
#                                "MMM", "TimeLost", "ClksThisMin", "LastICI_us",
#                                "ICI_rising", "avEndF", "MinF", "MaxF",
#                                "avNcyc", "MaxNcyc", "MaxSPL", "NofClstrs",
#                                "avClstrN.10", "Frange", "avSlope1",
#                                "avSlope2", "avBW",  "TrClassH", "TrClassM")]
# # run predicitions for each all models
# # boosted logistic regression
# predictions_lrb   <- caret::predict.train(fit_log_reg_boost,
#                                           newdata = all_data_tb,
#                                           type = "prob")
# predictions_gbm  <- caret::predict.train(gbmfit,
#                                          newdata = all_data_tb,
#                                          type = "prob")
# predictions_rf  <- caret::predict.train(fit_rf,
#                                         newdata = all_data_tb,
#                                         type = "prob")
# # during testing the use of the the probabilties and the thresholds seemed to give a better
# # result.
# # load the tresholds dataset
# thrs  <- t(read.csv("thresholds2.csv"))
# rownames(thrs)[2:4]  <- c("gbm", "rf", "lrb")
# predictions_lrb_class  <- ifelse(predictions_lrb$porp < thrs["lrb", 1], 0, 1)
# predictions_gbm_class  <- ifelse(predictions_gbm$porp < thrs["gbm", 1], 0, 1)
# predictions_rf_class  <- ifelse(predictions_rf$porp < thrs["rf", 1], 0, 1)
# predictions_df <- data.frame("lrb" = predictions_lrb_class,
#                              "gbm" = predictions_gbm_class,
#                              "rf" = predictions_rf_class)
# predictions_df$ensamble  <- apply(predictions_df, 1, Mode)
# #prev_pred  <- read.csv("pal-classifier\\data_with_final_predictions.csv")
# data_pred  <- cbind(all_data_tb, predictions_df)
# #data_pred  <- cbind(prev_pred, predictions_df)
# #remove the ensamble columns
# #colnames(data_pred)  <- make.names(colnames(data_pred), unique = TRUE)
# #data_pred  <- select(data_pred, -c(ensamble, ensamble.1))
# # data_pred2 <- data_pred   %>%
# #   mutate_at(c("lrb", "gbm", "rf", "ensamble"),
# #             \(x) ifelse(x == 0, 1, 0))
# # data_pred2$ensamble_final  <- apply(data_pred2[,c("lrb", "rf", "gbm", "lrb.1", "rf.1", "gbm.1")], 1, Mode)
# # create the density plot that louise created for the ensemble model for now
# # library(tidyr)
# # data_pred  %>% select(-c(lrb, rf, gbm)) %>%
# #   pivot_longer(-ensamble, names_to = "variables", values_to = "value")   %>%
# #   ggplot(aes(x = value, col = ensamble)) +
# #   geom_density() + facet_wrap(~variables, scales = "free")
# ## save the final data frame
# # colnames(data_pred2)[31:39] <- c("lrb", "gbm", "rf", "ensamble",
# #                                  "w_lrb", "w_gbm", "w_rf", "w_ensamble",
# #                                  "ensamble_final")
# write.csv(data_pred, "data_with_final_predictions_3.csv")
# endregion

# region for the test set of pal being on
# get all the files in the folder
all_files_porp <- list.files("ALLDATA\\test_set porps when pal on",
  pattern = "*.txt", full.names = TRUE
)
all_files_pals <- list.files("ALLDATA\\test_set pals when pal on",
  pattern = "*.txt", full.names = TRUE
)
all_data_porp <- lapply(all_files_porp, read.delim)
all_data_pal <- lapply(all_files_pals, read.delim)
# Join them together in a single dataframe
all_data_tb_porp <- do.call(rbind, all_data_porp)
all_data_tb_pal <- do.call(rbind, all_data_pal)
# add the classification column and merge
all_data_tb_porp$click <- "porp"
all_data_tb_pal$click <- "pal"
all_data_tb <- rbind(all_data_tb_pal, all_data_tb_porp)
clicks <- all_data_tb$click
all_data_tb <- preprocess_data(all_data_tb)
predictions_df <- run_predictions(all_data_tb, "thresholds2.csv")
cm <- confusionMatrix(
  as.factor(predictions_df$ensamble),
  as.factor(ifelse(clicks == "pal", 0, 1))
)

# confusion matrix
plt <- as.data.frame(cm$table)
plt_prop <- as.data.frame(prop.table(cm$table, 1))
plt_prop$Freq <- round(plt_prop$Freq * 100, 2)
plt$type <- "counts"
plt$fill <- scale(plt$Freq)
plt_prop$type <- "proportions"
plt_prop$fill <- scale(plt_prop$Freq)
plt <- rbind(plt, plt_prop)
plt$Prediction <- factor(plt$Prediction, levels = rev(levels(plt$Prediction)))
ggplot(plt, aes(Prediction, Reference, fill = fill)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "#009194") +
  labs(x = "Reference", y = "Prediction") +
  facet_wrap(~type, nrow = 1, scales = "free") +
  theme_minimal()

# histogram of wrong predictions
predictions_prob <- run_predictions(all_data_tb, "thresholds2.csv",
  return_probs = TRUE
)

predictions_prob <- lapply(
  names(predictions_prob),
  \(x) {
    predictions_prob[[x]]$model <- x
    return(predictions_prob[[x]])
  }
)
predictions_prob <- do.call(rbind, predictions_prob)
correct_detections <- predictions_df$lrb == ifelse(clicks == "pal", 0, 1)
predictions_prob$correct <- rep(correct_detections, 3)
predictions_prob$reference <- rep(clicks, 3)
# do a plot of the distribution of probabilities for all the models
ggplot(predictions_prob, aes(x = porp, fill = correct)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ model + reference, ncol = 2, scales = "free")

pca_of_things <- princomp(all_data_tb)
biplot(pca_of_things)
library(factoextra)
fviz_pca_var(pca_of_things,
  col.ind = as.factor(correct_detections)
)
fviz_contrib(pca_of_things,
  choice = "var", axes = 1:2, col.var = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)
# endregion
