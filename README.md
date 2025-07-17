# NFL Field Goal Success Probability Model

This project develops a logistic regression model to estimate the probability of a made field goal in the NFL, based on contextual and environmental features. 
It is designed to support analytics-informed coaching and front office decisions, such as:
- Whether to kick or go for it on 4th down
- Evaluating kicker performance across different conditions
- Informing roster or contract decisions
- Setting thresholds for field goal range confidence

## Model Summary
- Model Type: Logistic Regression
- Training Data: ~12,800 field goal attempts from NFL regular and postseason games (2010–2023)
- Testing Data: All field goal attempts from the 2024 season
- Target Variable: Binary outcome (FG made or missed)

### Key Features
Selected based on football intuition and domain expertise:
- Kick Distance (yards)
- Wind Speed (mph)
- Precipitation (binary)
- Last 2 Minutes of Half/Game (binary)
- Rookie Kicker (binary)
- Season
- Adjusted Kicker FG% (cumulative, regressed to league average)

### Performance Metrics (2024 Test Set)
Metric     | Value |
-----------|-------|
AUC	       | 0.771 |
Accuracy   | 0.822 |
Precision  | 0.888 |
Log Loss   | 0.351 |

These metrics reflect strong classification performance, well-calibrated probabilities, and high trustworthiness when used in live game scenarios.

**Why Logistic Regression?**
The model was selected for its balance of interpretability, performance, and calibration. While several models (lasso, random forest, weighted variants) were tested, the final “Football” model matched or outperformed more complex alternatives while remaining transparent and intuitive to explain.
- The baseline model used only kick distance and achieved an AUC of 0.774. By adding a handful of football-informed contextual features, the final model offers a more complete yet still parsimonious solution.

## Assumptions & Notes
- Blocked FGs were removed from training/testing data
- Kick direction and botched snaps/holds not modeled due to unavailable data
- Model assumes false positives (misses predicted as makes) are more costly than false negatives
- Kicker FG% is regressed to league average, weighted by sample size
- Model trained on post-2010 data due to structural FG success rate changes since early 2000s
- Data collected via nflfastR

## Next Steps
While the current model provides strong performance and interpretability, there are several ways it could be enhanced with additional data and refinement:
- Incorporate Kick Direction: Include left/right hash location once standardized data becomes available, which could help capture directional tendencies or kicker preferences.
- Wind Directionality: Expand weather features to capture wind relative to kick direction (e.g., headwind, tailwind, crosswind), which may further improve performance in marginal conditions.
- Special Teams Execution: Explore ways to include snap and hold quality, if reliable proxies become available (e.g., play-level tracking or video tags).
  
These additions would further improve the model’s accuracy and relevance, especially in high-leverage, weather-affected, or venue-specific scenarios.
