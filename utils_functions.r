## mode function
Mode <- function(x, na.rm = FALSE) {
  if (na.rm) {
    x <- x[!is.na(x)]
  }

  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# processing dataset
# this function performs the preprocessing of the dataset this includes things
# like one hot encoding and erasing the unnecesary columns.

preprocess_data <- function(data) {
  data$TrClassH <- ifelse(data$TrClass == "High", 1, 0)
  data$TrClassM <- ifelse(data$TrClass == "Mod", 1, 0)
  # selec the correct columns
  data <- data[, c(
    "Start", "TrDur_us",
    "NofClx", "nActualClx", "Clx.s", "ICIgood",
    "modalKHz", "avSPL", "MaxICI_us", "MinICI_us",
    "MMM", "TimeLost", "ClksThisMin", "LastICI_us",
    "ICI_rising", "avEndF", "MinF", "MaxF",
    "avNcyc", "MaxNcyc", "MaxSPL", "NofClstrs",
    "avClstrN.10", "Frange", "avSlope1",
    "avSlope2", "avBW", "TrClassH", "TrClassM"
  )]
  return(data)
}

run_predictions <- function(data, threshold_file, return_probs = FALSE) {
  predictions_lrb <- caret::predict.train(fit_log_reg_boost,
    newdata = data,
    type = "prob"
  )
  predictions_gbm <- caret::predict.train(gbmfit,
    newdata = data,
    type = "prob"
  )
  predictions_rf <- caret::predict.train(fit_rf,
    newdata = data,
    type = "prob"
  )
  thrs <- t(read.csv(threshold_file))
  rownames(thrs)[2:4] <- c("gbm", "rf", "lrb")
  predictions_lrb_class <- ifelse(predictions_lrb$porp < thrs["lrb", 1], 0, 1)
  predictions_gbm_class <- ifelse(predictions_gbm$porp < thrs["gbm", 1], 0, 1)
  predictions_rf_class <- ifelse(predictions_rf$porp < thrs["rf", 1], 0, 1)
  predictions_df <- data.frame(
    "lrb" = predictions_lrb_class,
    "gbm" = predictions_gbm_class,
    "rf" = predictions_rf_class
  )
  predictions_df$ensamble <- apply(predictions_df, 1, Mode)
  if (return_probs) {
    return(list(
      "lrb" = predictions_lrb,
      "gbm" = predictions_gbm,
      "rf" = predictions_rf
    ))
  } else {
    return(predictions_df)
  }
}
