# PAL_porp_classifier_scripts

## Description

This repositry provides a series of scripts to run a pretrained ML ensamble for classification between articial and natural porpoise calls detected by the CPOD (Cetacean and Porpoise Detector). 

## Context

Passive acoustic monitoring devices like the [CPOD](https://www.chelonia.co.uk/cpod.htm)  are essential for wildlife monitoring, allowing to collect large amounts of data. One of the key features of the CPOD is that the device does not store the acoustic data and instead process the detection on site, storing only a set of variables that describe the detected call. This allows the device to be highly efficient in terms of storage and energy.

In the recent years a new device called the [Porpoise Alert (PAL)](https://www.sciencedirect.com/science/article/pii/S0165783620302496?via%3Dihub) has gained popularity as a way to control porpoise bycatch from gillnets. However,  since PALs work by emitting a pre-recorded porpoise alert sound they can affect the accuracy of C-PODs, that will register this artificiall call as a natural porpoise.

To solve this problem we present a pretrained ensemble model that can accurately calssify observations as natural or artificial based on the variables stored by the CPOD. The model reached up to 95% accuracy during training and testing. Further processing is also available to deal with potential false negatives.

## Requirements and Installation

It is assumed that the user has basic understanding of R.

To use this script just download the entire repository. Don't change the internal structure of the files and dont change the default working directory.

All the scripts are written in R. The models are stored as R-objects from the package [caret](https://www.jstatsoft.org/article/view/v028i05). Other packages required are glue, plotROC, pROC, ggplot2, dplyr, caTools, gbm and randomForest

## Inference

Open the `1_Prediction.R` script 