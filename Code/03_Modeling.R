############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
###################################################### 3.0) MODELING #######################################################

source('02_EDA_Feature_Engineering.R')

head(pbp)

#-------------------------------------------------- TRAIN / TEST SPLIT --------------------------------------------------#

# Select relevant columns for modeling 
cols <- c(
  # General Game Information
  "season", "week", "spread_line", "total_line", "post_season",
  #️ Environmental Features
  "temp", "wind", "humidity", "rain", "snow", "precipitation", "freezing", "turf", "grass", "roof_closed", "high_altitude",
  # Play-Level Information
  "qtr", "quarter_seconds_remaining", "quarter_end", "half_end", "game_end", "drive", "yardline_100", "kick_distance",
  # Situational Features
  "score_differential", "timeout_prior", "last_two_minutes", "team_is_trailing", "tie_game", "timeouts_remaining",
  "kick_to_win", "kick_to_tie", "total_points", "prime_time", "at_home", "on_road",
  # Kicker-Specific Features
  "is_rookie", "adj_fg_pct", "adj_long_fg_pct",
  # Target Variable
  "fg_made"
)

set.seed(42)

# Split the data randomly into train and test sets -- train on prior seasons, test on 2024
train <- pbp %>% ungroup() %>% filter(season < 2024)
test <- pbp %>% ungroup() %>% filter(season == 2024)

train <- train %>% select(all_of(cols)) %>% drop_na()
test <- test %>% select(all_of(cols)) %>% drop_na()

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

# Function to plot a calibration curve
plot_calibration <- function(model, data, model_name = "Model") {
  # Predict field goal probabilities
  data$pred_prob <- predict(model, newdata = data, type = "response") 
  
  # Bin predictions into deciles
  calibration_data <- data %>%
    mutate(pred_bin = ntile(pred_prob, 10)) %>% 
    group_by(pred_bin) %>%
    summarise(
      avg_pred = mean(pred_prob, na.rm = TRUE),
      avg_actual = mean(fg_made, na.rm = TRUE),
      count = n()
    )
  
  # Create the plot
  calib_plot <- ggplot(calibration_data, aes(x = avg_pred, y = avg_actual)) +
    geom_line(color = "#0080C6", linewidth = 1) +
    geom_point(size = 2, color = "#0080C6") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(
      title = paste("Calibration Curve -", model_name),
      x = "Predicted FG Probability",
      y = "Actual FG%"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  return(calib_plot)
}

#----------------------------------------- MODELING: LOGISTIC REGRESSION -----------------------------------------#

######################## Baseline Logistic Regression with 1 Feature - Kick Distance ######################## 
baseline_model <- glm(fg_made ~ kick_distance, 
                      data = train, 
                      family = "binomial")

summary(baseline_model)

# Compute evaluation metrics
baseline_metrics <- compute_metrics(baseline_model, train, model_name = "baseline")
baseline_metrics

# Plot calibration curve
plot_calibration(baseline_model, train, model_name = "Baseline Model")

####################################### Full Model with All Features ###################################### 
full_model <- glm(fg_made ~ kick_distance + season + week + spread_line + total_line + post_season +
                    wind + temp + humidity + turf + roof_closed + precipitation + freezing + high_altitude +
                    qtr + quarter_seconds_remaining + drive + timeout_prior + last_two_minutes + score_differential + 
                    team_is_trailing + tie_game + timeouts_remaining + kick_to_win + kick_to_tie + total_points + 
                    prime_time + at_home + is_rookie + adj_fg_pct + adj_long_fg_pct,
                  data = train, 
                  family = "binomial")

summary(full_model)

# Compute evaluation metrics
full_model_metrics <- compute_metrics(baseline_model, train, model_name = "full_model")
full_model_metrics

# Plot calibration curve
plot_calibration(full_model, train, model_name = "Full Model")

################################## Model Based on Football Intuition #################################### 
football_model <- glm(fg_made ~ kick_distance + wind + precipitation + 
                        season + last_two_minutes + adj_fg_pct + is_rookie,
                      data = train, 
                      family = "binomial")


summary(football_model)

# Compute evaluation metrics
football_model_metrics <- compute_metrics(football_model, train, model_name = "football_model")
football_model_metrics

# Plot calibration curve
plot_calibration(football_model, train, model_name = "Football Features Model")

############################# Model with All Significant Features (p < 0.05) ############################# 
# Coincidentally very similar feature set to football intuition model features 
significant_model <- glm(fg_made ~ kick_distance + season + post_season + wind + 
                            + kick_to_win + is_rookie + adj_fg_pct,
                  data = train, 
                  family = "binomial")

summary(significant_model)

# Compute evaluation metrics
significant_model_metrics <- compute_metrics(significant_model, train, model_name = "significant_model")
significant_model_metrics

# Plot calibration curve
plot_calibration(significant_model, train, model_name = "Significant Features Model")

###################### Model Based on Football Intuition with Interaction Terms ###################### 
interaction_model <- glm(fg_made ~ kick_distance * wind + precipitation + last_two_minutes:kick_to_win +
                         is_rookie + adj_fg_pct,
                      data = train, 
                      family = "binomial")


summary(interaction_model)

# Compute evaluation metrics
interaction_model_metrics <- compute_metrics(interaction_model, train, model_name = "interaction_model")
interaction_model_metrics

# Plot calibration curve
plot_calibration(interaction_model, train, model_name = "Interactions Model")

############################## Model Based on Football Intuition with Class Weighting ################################ 
# Weight missed field goals slightly higher, as they only make up ~14% of the entire dataset, more harm in false positives
train$weight <- ifelse(train$fg_made == 0, 1.2, 1)
test$weight <- ifelse(test$fg_made == 0, 1.2, 1)

weighted_model <- glm(fg_made ~ kick_distance + wind + precipitation + 
                        season + last_two_minutes + adj_fg_pct + is_rookie,
                      data = train, 
                      weights = weight,
                      family = "binomial")


summary(weighted_model)

# Compute evaluation metrics
weighted_model_metrics <- compute_metrics(weighted_model, train, model_name = "weighted_model")
weighted_model_metrics

# Plot calibration curve
plot_calibration(weighted_model, train, model_name = "Class-Weighted Model")

########################################## LASSO Regression ######################################### 
# Create features matrix & target vector
X <- model.matrix(fg_made ~ kick_distance + season + week + post_season +
                    wind + temp + humidity + turf + roof_closed + precipitation + high_altitude +
                    qtr + timeout_prior + last_two_minutes + score_differential + 
                    timeouts_remaining + kick_to_win + is_rookie + adj_fg_pct,
                  data = train)[,-1]

y <- train$fg_made

# Run cross-validated Lasso
lasso_cv <- cv.glmnet(X, y, alpha = 1, family = "binomial")

# Check which features the lasso kept
coef(lasso_cv, s = "lambda.min")

# Predict FG probabilities
pred_prob <- as.vector(predict(lasso_cv, newx = X, type = "response", s = "lambda.min"))
pred_class <- ifelse(pred_prob >= 0.5, 1, 0)

# Compute AUC
lasso_auc <- auc(roc(y, pred_prob))

# Accuracy
lasso_accuracy <- mean(pred_class == y)

# Precision
tp <- sum(pred_class == 1 & y == 1)
fp <- sum(pred_class == 1 & y == 0)
lasso_precision <- ifelse((tp + fp) > 0, tp / (tp + fp), NA)

# Log Loss
epsilon <- 1e-15
pred_prob <- pmin(pmax(pred_prob, epsilon), 1 - epsilon)
lasso_log_loss <- -mean(y * log(pred_prob) + (1 - y) * log(1 - pred_prob))

# Return metrics in a dataframe
lasso_metrics <- data.frame(
  Model = "Lasso",
  AUC = round(lasso_auc, 3),
  Accuracy = round(lasso_accuracy, 3),
  Precision = round(lasso_precision, 3),
  Log_Loss = round(lasso_log_loss, 3)
)

lasso_metrics

# Plot calibration curve
calibration_data <- data.frame(
  fg_made = y,
  pred_prob = pred_prob
)

# Bin predictions into deciles and compute actual vs predicted means
calibration_summary <- calibration_data %>%
  mutate(pred_bin = ntile(pred_prob, 10)) %>%
  group_by(pred_bin) %>%
  summarise(
    avg_pred = mean(pred_prob, na.rm = TRUE),
    avg_actual = mean(fg_made, na.rm = TRUE),
    count = n()
  )

# Plot the calibration curve
lasso_calibration <- ggplot(calibration_summary, aes(x = avg_pred, y = avg_actual)) +
  geom_line(color = "#0080C6", linewidth = 1) +
  geom_point(size = 2, color = "#0080C6") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Calibration Curve - Lasso Model",
    x = "Predicted FG Probability",
    y = "Actual FG%"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

#--------------------------------------------- MODELING: RANDOM FOREST ---------------------------------------------#

# Convert fg_made target variable to a factor
train_rf <- train  
train_rf$fg_made <- as.factor(train_rf$fg_made)

# Random Forest with most features
rf_model <- randomForest(
  fg_made ~ kick_distance + season + week + wind + temp + turf + 
    roof_closed + precipitation + qtr + timeout_prior + score_differential + 
    timeouts_remaining + kick_to_win + is_rookie + adj_fg_pct, 
  data = train_rf,
  ntree = 500, 
  importance = TRUE
)

rf_model

# Make predictions
train_rf$pred_prob <- predict(rf_model, type = "prob")[, "1"]
train_rf$pred_class <- predict(rf_model, type = "response")  

# Compute AUC 
rf_auc <- auc(train_rf$fg_made, train_rf$pred_prob)

# Compute Accuracy & Precision
rf_accuracy <- mean(train_rf$pred_class == train_rf$fg_made)

tp <- sum(train_rf$pred_class == 1 & train_rf$fg_made == 1)
fp <- sum(train_rf$pred_class == 1 & train_rf$fg_made == 0)
rf_precision <- tp / (tp + fp)

# Compute Log Loss
epsilon <- 1e-15
rf_log_loss <- -mean(as.numeric(as.character(train_rf$fg_made)) * log(pmax(train_rf$pred_prob, epsilon)) +
                       (1 - as.numeric(as.character(train_rf$fg_made))) * log(pmax(1 - train_rf$pred_prob, epsilon)))

# Plot feature importance
varImpPlot(rf_model)

#--------------------------------------------- MODEL COMPARISONS ---------------------------------------------#

# Look at all evaluation metrics from all 7 models 
all_metrics <- rbind(baseline_metrics, 
      full_model_metrics, 
      significant_model_metrics, 
      football_model_metrics, 
      interaction_model_metrics,
      weighted_model_metrics,
      lasso_metrics) %>%
  rbind(data.frame(
    Model = "random_forest",
    AUC = round(rf_auc, 3),
    Accuracy = round(rf_accuracy, 3),
    Precision = round(rf_precision, 3),
    Log_Loss = round(rf_log_loss, 3)
  )) %>% 
  mutate(Model = str_to_title(gsub("_", " ", Model))) %>%
  arrange(-AUC, -Precision)

all_metrics

# Plot calibration curves for top-performing models 
plot_calibration(significant_model, train, model_name = "Significant Features Model")
plot_calibration(football_model, train, model_name = "Football Model")
plot_calibration(weighted_model, train, model_name = "Weighted Model")
lasso_calibration

# Champion Model = Football Model
# It’s nearly identical in AUC and other metrics as the stepwise and significant features models.
# It shows slightly better calibration in the high-confidence range.
# It uses fewer features, and all of the features included make sense based on football knowledge and can be explained to coaching staff.

#------------------------------------------- EVALUATING ON THE TEST SET ---------------------------------------------#

# Predict probabilities on test set
test$pred_prob <- predict(football_model, newdata = test, type = "response")

# Compute evaluation metrics
roc_test <- roc(test$fg_made, test$pred_prob)
auc_test <- auc(roc_test)

# Create realistic thresholds
thresholds <- seq(0.5, 0.8, by = 0.05)

# Compute accuracy, precision, and recall at various threshold
pr <- lapply(thresholds, function(t) {
  pred_class <- ifelse(test$pred_prob >= t, 1, 0)
  
  tp <- sum(pred_class == 1 & test$fg_made == 1)
  fp <- sum(pred_class == 1 & test$fg_made == 0)
  fn <- sum(pred_class == 0 & test$fg_made == 1)
  
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), NA)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), NA)
  
  data.frame(threshold = t, precision = precision, recall = recall)
}) %>% 
  bind_rows()

# Plot precision & recall at each threshold
ggplot(pr, aes(x = recall, y = precision)) +
  geom_line(color = "#0080C6", linewidth = 1) +
  geom_point(size = 3, color = "black") +
  geom_text(aes(label = round(threshold, 2)), vjust = 1.75, hjust = 1, size = 3.5) +
  labs(title = "Precision-Recall Curve with Thresholds",
       x = "Recall", y = "Precision") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Predicted class using 0.7 threshold
# 0.7 threshold keeps precision high, improves recall, remains a little conservative in predictions
test$pred_class <- ifelse(test$pred_prob >= 0.7, 1, 0)

# Compute other metrics
accuracy_test <- mean(test$pred_class == test$fg_made)

log_loss_test <- -mean(test$fg_made * log(pmax(test$pred_prob, 1e-15)) + 
                         (1 - test$fg_made) * log(pmax(1 - test$pred_prob, 1e-15)))

# Create a dataframe for test set metrics
test_metrics <- data.frame(
  Model = "Football Model",
  AUC = round(auc_test, 3),
  Accuracy = round(accuracy_test, 3),
  Precision = round(pr %>% filter(threshold == 0.7) %>% pull(precision), 3),
  Recall = round(pr %>% filter(threshold == 0.7) %>% pull(recall), 3),
  Log_Loss = round(log_loss_test, 3)
)

test_metrics

# Plot the calibration curve
plot_calibration(football_model, test, model_name = "Football Model (Test)")

# Check overall predicted vs actual FGs
sum(test$fg_made)
sum(test$pred_class)
mean(test$fg_made)
mean(test$pred_class)
mean(test$pred_prob)
mean(test$fg_made)

#------------------------------------------- FINAL MODEL TRAINING ---------------------------------------------#

# Train the final model on all available training data
all_data <- pbp

final_model <- glm(fg_made ~ kick_distance + wind + precipitation + 
                     season + last_two_minutes + adj_fg_pct + is_rookie,
                      data = all_data, 
                      family = "binomial")


summary(football_model)

# Predict probabilities for all field goals
all_data$fg_probability <- predict(football_model, newdata = all_data, type = "response")

# Predicted class (make vs miss) using 0.5 threshold
all_data$fg_made_prediction <- ifelse(all_data$fg_probability >= 0.7, 1, 0)

# Check overall predicted vs actual FGs - average fg_prediction = 86%
sum(all_data$fg_made)
sum(all_data$fg_made_prediction)
mean(all_data$fg_made)
mean(all_data$fg_made_prediction)
mean(all_data$fg_probability)
