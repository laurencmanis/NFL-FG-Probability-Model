############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
################################################ 0.0) LIBRARIES & FUNCTIONS ################################################# 
library(dplyr)
library(tidyverse)
library(ggplot2)
library(nflfastR)
library(stringr)
library(glmnet)
library(caret)
library(lubridate)
library(scales)
library(leaps)
library(ggtext)
library(pROC)
library(randomForest)
library(car)
library(yardstick)

#------------------------------------------------ RECURRING FUNCTIONS ------------------------------------------------#

# Function to compute evaluation metrics 
compute_metrics <- function(model, data, model_name = "model") {
  # Predict field goal probabilities
  data$pred_prob <- predict(model, newdata = data, type = "response") 
  
  # Compute AUC
  roc <- roc(data$fg_made, data$pred_prob)
  auc <- auc(roc)
  
  # Compute Accuracy & Precision
  data$pred_class <- ifelse(data$pred_prob >= 0.5, 1, 0)
  accuracy <- mean(data$pred_class == data$fg_made)
  
  tp <- sum(data$pred_class == 1 & data$fg_made == 1)
  fp <- sum(data$pred_class == 1 & data$fg_made == 0)
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), NA)
  
  # Compute Log Loss
  epsilon <- 1e-15 
  log_loss <- -mean(data$fg_made * log(pmax(data$pred_prob, epsilon)) +
                      (1 - data$fg_made) * log(pmax(1 - data$pred_prob, epsilon)))
  
  # Return a data frame of metrics
  return(data.frame(
    Model = model_name,
    AUC = round(auc, 3),
    Accuracy = round(accuracy, 3),
    Precision = round(precision, 3),
    Log_Loss = round(log_loss, 3)
  ))
}

# Function to plot binned calibration, using 5-yard buckets
plot_calibration <- function(model, data, model_name = "Model") {
  # Predict field goal probabilities
  data$pred_prob <- predict(model, newdata = data, type = "response")
  
  # Create field goal distance buckets
  data <- data %>%
    mutate(
      distance_bucket = case_when(
        kick_distance < 25 ~ "<25",
        kick_distance >= 25 & kick_distance < 30 ~ "25-29",
        kick_distance >= 30 & kick_distance < 35 ~ "30-34",
        kick_distance >= 35 & kick_distance < 40 ~ "35-39",
        kick_distance >= 40 & kick_distance < 45 ~ "40-44",
        kick_distance >= 45 & kick_distance < 50 ~ "45-49",
        kick_distance >= 50 & kick_distance < 55 ~ "50-54",
        kick_distance >= 55 ~ "55+"
      )
    )
  
  # Summarize calibration by distance bucket
  calibration_data <- data %>%
    group_by(distance_bucket) %>%
    summarise(
      avg_pred = mean(pred_prob, na.rm = TRUE),
      avg_actual = mean(fg_made, na.rm = TRUE),
      count = n()
    ) %>%
    ungroup() %>%
    mutate(
      distance_bucket = factor(
        distance_bucket,
        levels = c("<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55+")
      ))
  
  # Plot calibration curve
  calib_plot <- ggplot(calibration_data, aes(x = avg_pred, y = avg_actual)) +
    geom_line(group = 0.9, color = "#0080C6", linewidth = 1) +
    geom_point(size = 2, color = "#0080C6") +
    geom_text(aes(label = distance_bucket), nudge_x = -0.035, size = 3.5, fontface = "bold") + 
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(
      title = paste("Calibration Curve by Distance -", model_name),
      x = "Average Predicted FG Probability",
      y = "Actual FG %"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
    scale_x_continuous(limits = c(0.45, 1), breaks = seq(0.4, 1, 0.1))
  
  return(calib_plot)
}

