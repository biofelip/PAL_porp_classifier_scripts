# This script takes raw data from a folder and produces a dataframe with
# the raw data and the predicted class for each model and one ensmeble
# predicition.
library(caret)
library(dplyr)
library(caTools)
library(gbm)
library(randomForest)
library(here)
here()
source("utils_functions.r")
# Load the three models
load("log_reg_boosted_gridsearch2.RData")
load("rf_gridsearch2.RData")
load("gbmfit2.RData")
#
all_data_tb <- read.csv("your_CPOD_data.csv")
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
