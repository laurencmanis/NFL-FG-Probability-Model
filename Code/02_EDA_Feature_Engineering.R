############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
############################################# 2.0) EDA & FEATURE ENGINEERING ###############################################

source('01_Data_Preparation.R')

#------------------------------------------------------ INITIAL EDA ------------------------------------------------------#

# Average FG Pct by season
pbp %>% group_by(season) %>% summarise(mean(fg_made))

# Plot average FG percent by season - Hypothesis: kickers have improved over the years
avg_fg_by_season <- pbp %>%
  group_by(season) %>%
  summarise(fg_pct = mean(fg_made, na.rm = TRUE)) %>%
  ggplot(aes(x = season, y = fg_pct)) +
  geom_line(color = "#0080C6", linewidth = 1) +
  geom_point(color = "#0080C6", size = 2) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Field Goal Percentage by Season",
    x = "Season",
    y = "FG%"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold")
  ) +
  scale_x_continuous(breaks = seq(2000, 2025, 2)) 

# Clear upward trend until 2008, dramatic drop in 2009, then >=83% through 2024, consistently >=82% since 2004
avg_fg_by_season

# FG% before 2010 = 82%, FG% since 2010 = 86%
pbp %>% 
  mutate(bucket = if_else(season < 2010, '2000-2009', '2010-2024')) %>% 
  group_by(bucket) %>%
  summarise(fg_pct = mean(fg_made, na.rm = TRUE)) 

# Drop all attempts before 2010 season - do not want to train on outdated information
pbp <- pbp %>% filter(season >= 2010)

# Total number of field goals made & attempted
sum(pbp$fg_made)
nrow(pbp)

# Proportion of field goals made 
mean(pbp$fg_made)
table(pbp$fg_made)
prop.table(table(pbp$fg_made))

# Look at total number of attempts by distance 
pbp %>%
  group_by(kick_distance) %>%
  summarise(n_kicks = n()) %>% 
  ggplot(aes(x = kick_distance, y = n_kicks)) +
  geom_bar(stat = "identity", fill = "#0080C6") +
  labs(
    title = "Total FG Attempts by Distance",
    x = "Kick Distance",
    y = "Attempts"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold")
  ) +
  scale_x_continuous(breaks = seq(10, 70, 5)) 

# Create buckets and plot total number of attempts by distance bucket
pbp %>%
  mutate(bucket = case_when(
    kick_distance < 25 ~ '<25',
    kick_distance < 30 ~ '25-29',
    kick_distance < 35 ~ '30-34',
    kick_distance < 40 ~ '35-39',
    kick_distance < 45 ~ '40-44',
    kick_distance < 50 ~ '45-49',
    kick_distance < 55 ~ '50-54',
    TRUE ~ '55+'
  )) %>%
  group_by(bucket) %>%
  summarise(n_kicks = n()) %>% 
  ggplot(aes(x = bucket, y = n_kicks)) +
  geom_bar(stat = "identity", fill = "#0080C6") +
  labs(
    title = "Total FG Attempts by Distance",
    x = "Kick Distance",
    y = "Attempts"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold")
  )

# The sample of field goal attempts from 55+ yards is not only small but also inherently biased. 
# In practice, coaches are unlikely to send out a weaker kicker for a long attempt, especially when other options exist. 
# As a result, these attempts are disproportionately taken by stronger kickers, which inflates the observed make rate 
# at those distances compared to the true, population-level probability of success for any kicker.

#----------------------------------------------- SAMPLE SIZE & SYNTHETIC MISSES ----------------------------------------------#

# Generate synthetic field goal attempts to account for selection bias:
  # Coaches are unlikely to attempt 55+ yard field goals with weaker kickers, even in favorable conditions — they often choose to punt instead.
  # As a result, observed data at long distances overrepresents strong kickers and overstates average make probability.

# To simulate more realistic scenarios, we:
  # - Identify 4th-down punts in good weather from 55-65 yard field goal range.
  # - Replace these with synthetic missed field goals.
  # - Randomly sample ~2% of the full dataset to inject as noise, maintaining realism.

set.seed(123)

synthetic <- data %>% 
  ungroup() %>%
  filter(
    season >= 2010 &
    # 4th downs where coaches chose to punt, rather than attempt a FG or go for it
    down == 4 & play_type == 'punt' & 
    # Favorable kicking conditions 
    temp >= 50 & wind < 10 & precipitation == 0 & high_altitude == 0 &
    # Would-be kick distance 55-65 yards
    yardline_100 >= 37 & yardline_100 < 47) %>%
  mutate(
    # Flip the play info and outcome 
    play_type = 'field_goal',
    field_goal_attempt = 1,
    field_goal_result = 'missed',
    fg_made = 0) %>%
  select(-kicker_player_name, -kicker_player_id) %>%
  distinct()

# Identify which kickers would have been sent out in the above scenarios
ks <- data %>%
  filter(play_type %in% c("extra_point", "field_goal") & !is.na(kicker_player_id)) %>%
  group_by(season, game_id, posteam, kicker_player_id, kicker_player_name) %>%
  summarise(kicks = n(), .groups = "drop") %>%
  group_by(season, game_id, posteam) %>%
  slice_max(kicks, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(-kicks)

synthetic <- synthetic %>%
  left_join(ks, by = c("season", "game_id", "posteam"))


#-------------------------------------------------- FEATURE ENGINEERING --------------------------------------------------#

# Create an indicator if a kicker is a rookie (assumption that rookies have played in 0 prior seasons)
# First season for each kicker
kicker_first_season <- pbp %>%
  filter(!is.na(kicker_player_id)) %>%
  group_by(kicker_player_name, kicker_player_id) %>%
  summarise(first_season = min(season)) %>%
  ungroup()

# Join back and create rookie flag
pbp <- pbp %>%
  left_join(kicker_first_season, by = c("kicker_player_name","kicker_player_id")) %>%
  mutate(is_rookie = if_else(season == first_season, 1, 0))

# Compute cumulative in-season kicking stats for each kicker, as of, but excluding each game
kicker_cume_stats <- pbp %>%
  # Compute total field goals attempted and made for each game
  group_by(kicker_player_id, kicker_player_name, season, week, game_id) %>%
  summarise(
    fg_attempts = sum(field_goal_attempt),
    fg_makes = sum(fg_made),
    long_attempts = sum(long_kick),
    long_makes = sum(long_kick * fg_made)
  ) %>%
  ungroup() %>%
  # Compute cumulative stats for each kicker
  arrange(kicker_player_id, season, week) %>%
  group_by(kicker_player_id, kicker_player_name, season) %>%
  mutate(
    # Total FG makes, attempts, rate
    cum_fg_made = lag(cumsum(fg_makes), default = 0),
    cum_fg_attempts = lag(cumsum(fg_attempts), default = 0),
    cum_fg_pct = if_else(cum_fg_attempts > 0, cum_fg_made / cum_fg_attempts, NA_real_),
    # Long (50+ Yard) FG makes, attempts, rate
    cum_long_fg_made = lag(cumsum(long_makes), default = 0),
    cum_long_fg_attempts = lag(cumsum(long_attempts), default = 0),
    cum_long_fg_pct = if_else(cum_long_fg_attempts > 0, cum_long_fg_made / cum_long_fg_attempts, NA_real_)
  ) %>%
  distinct(game_id, kicker_player_id, kicker_player_name, season, week,
           cum_fg_attempts, cum_fg_pct, cum_long_fg_attempts, cum_long_fg_pct) %>%
  ungroup()

# Compute league-average cumulative stats, in order to regress kicker stats to the mean and better account for early weeks, low-volume of kicks, etc.
league_cume_fg <- pbp %>%
  group_by(season, week) %>%
  summarise(
    fg_makes = sum(fg_made),
    fg_attempts = n(),
    long_fg_makes = sum(long_kick * fg_made),
    long_fg_attempts = sum(long_kick)
  ) %>%
  arrange(season, week) %>%
  mutate(
    # Compute cumulative field goals made and attempted
    cum_fg_makes = lag(cumsum(fg_makes), default = 0),
    cum_fg_attempts = lag(cumsum(fg_attempts), default = 0),
    cum_long_fg_makes = lag(cumsum(long_fg_makes), default = 0),
    cum_long_fg_attempts = lag(cumsum(long_fg_attempts), default = 0),
    # Compute rolling league average field goal percentages 
    league_fg_pct = if_else(cum_fg_attempts > 0, cum_fg_makes / cum_fg_attempts, NA_real_),
    league_long_fg_pct = if_else(cum_long_fg_attempts > 0, cum_long_fg_makes / cum_long_fg_attempts, NA_real_)
  ) %>%
  select(season, week, league_fg_pct, league_long_fg_pct)

# Join league averages back to kicker cumulative stats
kicker_cume_stats <- kicker_cume_stats %>%
  left_join(league_cume_fg, by = c("season", "week"))

# Compute average cumulative attempts per kicker by week
avg_kicker_attempts <- pbp %>%
  group_by(season, week, kicker_player_id) %>%
  # Get kicker stats per game (week)
  summarise(
    attempts = n(),
    makes = sum(fg_made),
    long_attempts = sum(long_kick),
    long_makes = sum(long_kick * fg_made)
  ) %>%
  # Aggregate to average stats across all kickers in each week/season
  group_by(season, week) %>%
  summarise(
    avg_kicker_attempts = mean(attempts),
    avg_kicker_makes = mean(makes),
    avg_fg_pct = sum(makes) / sum(attempts),
    avg_kicker_long_attempts = mean(long_attempts),
    avg_long_fg_pct = if_else(sum(long_attempts) > 0, sum(long_makes) / sum(long_attempts), NA_real_)
  )

# Join league averages back to kicker cumulative stats
kicker_cume_stats <- kicker_cume_stats %>%
  left_join(avg_kicker_attempts, by = c("season", "week"))

# For games in week 1, assume every kicker is average, impute the prior season's average FG%
avg_fg_pct <- avg_kicker_attempts %>%
  group_by(season) %>%
  summarise(
    attempts = sum(avg_kicker_attempts),
    makes = sum(avg_kicker_makes),
    avg_fg_pct = makes / attempts
  )

kicker_cume_stats <- kicker_cume_stats %>%
  left_join(avg_fg_pct %>% rename(prior_season = season, prior_season_fg_pct = avg_fg_pct),
            by = c("season" = "prior_season")) %>%
  mutate(
    cum_fg_pct = if_else(week == 1, prior_season_fg_pct, cum_fg_pct),
    cum_fg_attempts = if_else(week == 1, 0, cum_fg_attempts)
  )

# Apply shrinkage based on a dynamic prior, where prior is the league average attempts
## Early in the season: All kickers are uncertain/have few attempts → shrink more
## Later in the season: Most kickers are well-known/have many attempts → shrink less unless, someone is low-volume
kicker_cume_stats <- kicker_cume_stats %>%
  mutate(
    adj_fg_pct = (avg_fg_pct * avg_kicker_attempts + cum_fg_pct * cum_fg_attempts) /
      (avg_kicker_attempts + cum_fg_attempts),
    adj_long_fg_pct = (avg_long_fg_pct * avg_kicker_long_attempts + cum_long_fg_pct * cum_long_fg_attempts) /
      (avg_kicker_long_attempts + cum_long_fg_attempts)
  ) %>%
  replace(is.na(.), 0)

# Join kicker cumulative stats back to play-by-play data
pbp <- pbp %>%
  left_join(kicker_cume_stats %>%
              select(game_id, kicker_player_id, kicker_player_name, season, week, 
                     cum_fg_attempts, cum_long_fg_attempts, adj_fg_pct, adj_long_fg_pct), 
            by=c("game_id","kicker_player_id","kicker_player_name","season","week")) %>% 
  ungroup()

sum(pbp$field_goal_attempt)
head(pbp)


#--------------------------------------------------------- EDA ---------------------------------------------------------#

# Plot categorical variables against average field goal percentage to observe trends 
cat_vars <- c("location", "timeout_prior", "rain", "snow", "precipitation", "freezing", "roof_closed", 
              "high_altitude", "turf",  "grass", "post_season", "last_two_minutes", "team_is_trailing", 
              "tie_game","slate", "prime_time", "kick_to_win", "kick_to_tie", "is_rookie","at_home","on_road")

# Create and store plots in a named list
cat_plots <- list()

for (var in cat_vars) {
  plot_data <- pbp %>%
    group_by(.data[[var]]) %>%
    summarise(fg_pct = mean(fg_made, na.rm = TRUE)) %>%
    mutate(label = paste0(round(fg_pct * 100, 1), "%"))
  
  p <- ggplot(plot_data, aes_string(x = var, y = "fg_pct")) +
    geom_col(fill = "#0080C6") +
    geom_text(aes(label = label), vjust = -0.5, size = 4) +
    labs(
      title = paste("FG% by", str_to_title(gsub("_", " ", var))),
      x = var,
      y = "FG%"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5)
    ) +
    ylim(0, 1)
  
  cat_plots[[var]] <- p
}

cat_plots[2]
cat_plots[3]
cat_plots[4]
cat_plots[5]
cat_plots[10]
cat_plots[11]
cat_plots[16]
cat_plots[17]
cat_plots[18]

# Observations
# Post-Timeout: FG% is lower on attempts that occur immediately following a timeout, suggesting a potential "icing" effect.
# Rain: FG% is noticeably lower in rainy conditions, highlighting the importance of weather on kick outcomes.
# Snow: FG% is lower in snowy conditions as well
# Precipitation: FG% is lower when there is any form of precipitation, when the field and ball are likely wet and slippery
# Post-Season: Kicks attempted during postseason games show a higher FG%, potentially reflecting a combination of higher kicker quality, or more conservative play-calling.
# End-of-Half/Game: FG% drops in the final two minutes of a half or game, likely due to increased pressure or longer kick distances in end-of-clock scenarios.
# Kick to Win: FG% drops dramatically when the kick is to win the game, potentially due to increased pressure, or longer distances  
# Kick to Tie: FG% drops dramatically when the kick is to tie the game and force OT
# Rookie: FG% is lower for rookie kickers, as they have less experience


# Plot numeric/continuous variables against average field goal percentage to observe trends 
cont_vars <- c("season","week","spread_line","total_line","wind","temp","qtr","quarter_seconds_remaining",
               "score_differential","total_points", "kick_distance", "humidity","timeouts_remaining","cum_fg_attempts")

# Create and store scatter plots in a named list
cont_plots <- list()

for (var in cont_vars) {
  plot_data <- pbp %>%
    filter(!is.na(.data[[var]])) %>%
    group_by(.data[[var]]) %>%
    summarise(
      fg_pct = mean(fg_made, na.rm = TRUE)
    ) %>%
    mutate(label = paste0(round(fg_pct * 100, 1), "%"))
  
  p <- ggplot(plot_data, aes_string(x = var, y = "fg_pct")) +
    geom_point(color = "#0080C6", size = 2) +
    labs(
      title = paste("FG% by", str_to_title(gsub("_", " ", var))),
      x = var,
      y = "FG%"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5)
    ) +
    ylim(0.7, 1)
  
  cont_plots[[var]] <- p
}

cont_plots[2]
cont_plots[5]
cont_plots[11]
cont_plots[13]
cont_plots[14]

# Observations
# Week-In-Season: FG% tends to increase as the season progresses, possibly reflecting improved kicker rhythm, or selection bias (i.e., struggling kickers being replaced).
# Wind: There is a slight negative relationship between wind speed and FG%, consistent with the expectation that stronger winds increase kick difficulty.
# Kick Distance: As expected, there is a strong negative relationship between kick distance and FG%, confirming that longer kicks are significantly harder to convert.
# Timeouts Remaining: FG% increases with the number of timeouts remaining for the kicking team, potentially reflecting improved play-calling flexibility or reduced pressure.
# Cumulative Attempts: FG% increases with the number of attempts the kicker has taken during the season to date

# Plot the distributions of continuous variables to identify any heavy skews, abnormalities that may need to be handled
dist_plots <- list()

for (var in cont_vars) {
  p <- ggplot(pbp, aes_string(x = var)) +
    geom_histogram(fill = "#0080C6", color = 'black', bins = 15, alpha = 0.8) +
    labs(
      title = paste("Distribution of", str_to_title(gsub("_", " ", var))),
      x = var,
      y = "Count"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
  
  dist_plots[[var]] <- p
}

dist_plots[5]
dist_plots[6]
dist_plots[8]
dist_plots[10]

# Observations
# Wind: Wind is heavily right-skewed, with the majority of kicks occurring in low-wind conditions, very few attempts occuring in wind speeds exceeding 20 mph.
# Temperature: Temperature shows a slight left skew, with relatively fewer kicks occurring in freezing conditions.
# Seconds Remaining in Quarter: There is an outsized number of field goal attempts with virtually no time remaining in the quarter, likely due to intentional clock management at the end of halves.
# Total Points: The distribution of total combined points at the time of the kick is right-skewed, with fewer kicks attempted when the score is especially high (e.g., ≥ 50 points).

# Knowing that distance will be the primary predictor, look further into its distribution
summary(pbp$kick_distance)
sum(pbp$kick_distance > 60)
sum(pbp$kick_distance > 55)
sd(pbp$kick_distance)

# Explore some potential interactions of features, based on football context
# Kick Distance & Wind
ggplot(pbp, aes(x = kick_distance, y = wind, color = factor(fg_made))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(
    values = c("0" = "red", "1" = "seagreen"), 
    labels = c("Missed", "Made")
  ) +
  labs(title = "FG Made by Kick Distance and Wind", color = "FG Made") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = 'top')

# Wind may amplify the kick difficulty at longer distances

# Kick Distance & Temperature
ggplot(pbp, aes(x = kick_distance, y = temp, color = factor(fg_made))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(
    values = c("0" = "red", "1" = "seagreen"), 
    labels = c("Missed", "Made")
  ) +
  labs(title = "FG Made by Kick Distance and Temperature", color = "FG Made") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = 'top')

# Temperature effects are more pronounced at longer distances


#---------------------------------------------- INJECTING SYNTHETIC MISSES -----------------------------------------------#

# Fill in kicker values, based on the assumed kicker's most recent stats prior to the synthetic attempt
synthetic <- synthetic %>%
  left_join(kicker_cume_stats %>%
              select(game_id, kicker_player_id, kicker_player_name, season, week, 
                     cum_fg_attempts, cum_long_fg_attempts, adj_fg_pct, adj_long_fg_pct), 
            by=c("game_id","kicker_player_id","kicker_player_name","season","week")) %>% 
  ungroup()

# First season for each kicker
kicker_first_season <- data %>%
  filter(!is.na(kicker_player_id)) %>%
  group_by(kicker_player_name, kicker_player_id) %>%
  summarise(first_season = min(season)) %>%
  ungroup()

# Join back and create rookie flag
synthetic <- synthetic %>%
  left_join(kicker_first_season, by = c("kicker_player_name","kicker_player_id")) %>%
  mutate(is_rookie = if_else(season == first_season, 1, 0))

synthetic <- synthetic %>% filter(!is.na(adj_fg_pct))

# Randomly select a small sample of synthetic rows to use in training - equal to 2% of total observations
n_syn <- round(nrow(pbp) * 0.02)

synthetic <- synthetic %>%
  sample_n(n_syn) %>%
  mutate(synthetic = TRUE)

pbp <- pbp %>% mutate(synthetic = FALSE)

head(synthetic)
