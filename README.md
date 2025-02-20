# PAL_porp_classifier_scripts

## Description

This repositry provides a series of scripts to run a pretrained ML ensamble for classification between artificial and natural porpoise calls detected by the CPOD (Cetacean and Porpoise Detector).

## Context

Passive acoustic monitoring devices like the [CPOD](https://www.chelonia.co.uk/cpod.htm)  are essential for wildlife monitoring, allowing to collect large amounts of data. One of the key features of the CPOD is that the device does not store the acoustic data and instead process the detection on site, storing only a set of variables that describe the detected call. This allows the device to be highly efficient in terms of storage and energy.

In the recent years a new device called the Porpoise Alert [(PAL)](https://www.sciencedirect.com/science/article/pii/S0165783620302496?via%3Dihub) has gained popularity as a way to control porpoise bycatch from gillnets. However,  since PALs work by emitting a pre-recorded porpoise alert sound they can affect the accuracy of C-PODs, that will register this artificiall call as a natural porpoise.

To solve this problem we present a pretrained ensemble model that can accurately calssify observations as natural or artificial based on the variables stored by the CPOD. The model reached up to 95% accuracy during training and testing. Further processing is also available to deal with potential false negatives.

## Requirements and "Installation"

It is assumed that the user has basic understanding of R.

To use this script just download the entire repository. Make sure that root of this repository is your working directory, in other words `getwd()` should return something like `"C:\Path_to_folder\PAL_porp_classifier_scripts"`.

All the scripts are written in R. The models are stored as R-objects from the package [caret](https://www.jstatsoft.org/article/view/v028i05). Other packages required are glue, plotROC, pROC, ggplot2, dplyr, caTools, gbm, randomForest and butcher

## Inference

The repositorty contains our trained model objects and they can be used without further modification (later sections will show how to train your own model) the models sizes have been reduced using the package [butcher](https://butcher.tidymodels.org/) to fit into this repository. Open the `1_Prediction.R` script, this is a simple script that run our ensemble using some custom functions.

The infrence  is carried out by a main function `run_predictions` and other auxilary functions. For these functions to work make sure that you source the `utils_functions.r` script and load the pretrained models.

```{r}
library(caret)
library(dplyr)
library(caTools)
library(gbm)
library(randomForest)
# source the functions
source("utils_functions.r")
# Load the three models
load("log_reg_boosted_gridsearch2.RData")
load("rf_gridsearch2.RData")
load("gbmfit2.RData")
# load your data
all_data_tb <- read.csv("your_CPOD_data.csv")

```

 `run_predictions` first does some  preprocessing  including checking that the relevant columns are present in the dataset, and one hot encoding of the factor variables. This function has three arguments, the  data frame to run the inference in, a csv with the thresholds (provided in this repository) for the models and boolean argument to indicate if probabilities or the prediction categories should be returned.

```{r}
predictions_df <- run_predictions(
  data = all_data_tb,
  threshold_file = "thresholds2.csv",
  return_probs = "FALSE"
)
```

The resulting dataframe contains the same number of rows as the original data and four colums that correspond to the predictions for each model and a fourth column for the ensamble prediction (remember than 0 means PAL and 1 means a porpoise call).

| lrb| gbm| rf| ensamble|
|:---:|:---:|:--:|:--------:|
| 1  | 0  | 1 |   1     |
| 1  | 1  | 1 |   1     |
| 1  | 1  | 1 |   1     |
| 0  | 1  | 1 |   1     |
| 1  | 0  | 0 |   0     |
| 1  | 1  | 1 |   1     |

We can combine this prediction df  with the original dataframe and perform some post-processing.
Despite their very high accuracy, our models (as with any ML model) can still make some errors, this specially true for the PALs. Because of this we deviced a method to erase PAL detections that are very likely false negatives.

The method is based on the notion that PALs are coming from glints that are placed for several hours in a row. We want to ensure that signals that were  assigned to PALs only remain as PAL if PAL signals occur over at least two consecutive hours. Otherwise, they should be corrected to Porpoises.

We wrote a function called `post_processing` that carries out this step. For runing this function we need to combine our predictions with the originall data frame

```{r}
# combine the original data with predictions
data_and_predictions <- bind_cols(all_data_tb, predictions_df) 
data_and_predictions <- post_processing(data_and_predictions)
# compare the results
corr_comp <- with(
  data_and_predictions,
  cbind(table(ensamble), table(corrected_category))
)
colnames(corr_comp) <- c("Before Correction", "After Correction")
rownames(corr_comp) <- c("PAL", "porp")
corr_comp


```

As can be seen the function adds a new column called `corrected_category` that removes the PAL detections that ocurred on a single hour.

|Var |  Before correction|After correction|
|:----:|:-----:|:-----:|
|PAL    |   566|240|
|porp    | 39988|40314|

Finally `run_predictions` can also return the raw probabilites for ech observation when  `return_prob = TRUE` then the function returns a list of lenght three with each element of the list being a dataframe with two columns containig the probability for each category.

```{r}
predictions_df <- run_predictions(
  data = all_data_tb,
  threshold_file = "thresholds2.csv",
  return_probs = "FALSE"
)
predictions_df$gbm |> head()
```

|    pal    |   porp    |
|:---------:|:---------:|
| 0.0179862 | 0.9820138 |
| 0.0179862 | 0.9820138 |
| 0.1192029 | 0.8807971 |
| 0.0000001 | 0.9999999 |
| 0.0000008 | 0.9999992 |
| 0.0000000 | 1.0000000 |

