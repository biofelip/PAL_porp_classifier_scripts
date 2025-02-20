proper_col_names <- c(
  "Start", "TrDur_us", "NofClx", "nActualClx",
  "Clx.s", "ICIgood", "modalKHz", "avSPL", "MaxICI_us",
  "MinICI_us", "MMM", "TimeLost", "ClksThisMin",
  "LastICI_us", "ICI_rising", "avEndF", "MinF",
  "MaxF", "avNcyc", "MaxNcyc", "MaxSPL", "NofClstrs",
  "avClstrN.10", "Frange", "avSlope1", "avSlope2",
  "avBW", "TrClass"
)
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
  if (!(all(proper_col_names %in% colnames(data)))) {
    stop(paste("The dataframe does not contain the proper columns type, the dataframe should contain:", proper_col_names))
  }
  data$TrClassH <- ifelse(data$TrClass == "High", 1, 0)
  data$TrClassM <- ifelse(data$TrClass == "Mod", 1, 0)

  return(data)
}

run_predictions <- function(data, threshold_file, return_probs = FALSE) {
  data <- preprocess_data(data)
  predictions_lrb <- caret::predict.train(fit_log_reg_boost,
    newdata = data,
    type = "prob"
  )
  predictions_gbm <- caret::predict.train(fit_gbm,
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

plot_cf <- function(cf) {
  plt <- as.data.frame(cf$table)
  plt_prop <- as.data.frame(prop.table(cf$table, 1))
  plt_prop$Freq <- round(plt_prop$Freq * 100, 2)
  plt$type <- "counts"
  plt$fill <- scale(plt$Freq)
  plt_prop$type <- "proportions"
  plt_prop$fill <- scale(plt_prop$Freq)
  plt <- rbind(plt, plt_prop)
  plt$Prediction <- factor(plt$Prediction, levels = rev(levels(plt$Prediction)))
  ggplot2::ggplot(plt, ggplot2::aes(Prediction, Reference, fill = fill)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(aes(label = Freq)) +
    ggplot2::scale_fill_gradient(low = "white", high = "#009194") +
    ggplot2::labs(x = "Reference", y = "Prediction") +
    ggplot2::facet_wrap(~type, nrow = 1, scales = "free") +
    ggplot2::theme_minimal()
}

### post processing jefff
post_processing <- function(data) {
  if (!(all(c("ensamble", "Time") %in% colnames(data)))) {
    stop("The dataframe must contain a Time column and an ensamble column,\n
         Remember to join the predictions df with the original dataframe")
  }
  data$datetime <- lubridate::floor_date(lubridate::dmy_hm(data$Time),
    unit = "hour"
  )
  # Check whether PALS were recorded in consecutive hours
  data_PAL_check <- data |>
    dplyr::filter(ensamble == 0) |>
    dplyr::arrange(datetime) |>
    dplyr::group_by(datetime) |>
    dplyr::summarise(count = dplyr::n(), .groups = "drop") |>
    dplyr::mutate(next_hour = dplyr::lead(datetime)) |>
    dplyr::mutate(time_diff = difftime(next_hour, datetime, units = "hours")) |>
    dplyr::filter(time_diff != 1 & count < 10)
  # Correct the observations that where just recorded in one hour and not in consecutive hours
  data_c <- data |>
    dplyr::rowwise() |> # Work rowwise
    dplyr::mutate(
      corrected_category = dplyr::if_else(
        any(abs(difftime(datetime, data_PAL_check$datetime, units = "hours")) <= 1), # If within 1 hour
        1, # change to porp
        ensamble # If not, porp/pal categorisation remains unchanged
      )
    ) |>
    dplyr::ungroup()
  return(data_c)
}
