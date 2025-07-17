############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
############################################# 2.0) EDA & FEATURE ENGINEERING ###############################################

source('01_Data_Preparation.R')

#-------------------------------------------------- FEATURE ENGINEERING --------------------------------------------------#

# Binarize Field Goal Information & Create Target Variable
pbp <- pbp %>%
  mutate(fg_made = if_else(field_goal_result == 'made', 1, 0),
         fg_missed = if_else(field_goal_result != 'made', 1, 0))

# Total number of field goals made 
sum(pbp$fg_made)

# Proportion of field goals made 
mean(pbp$fg_made)
table(pbp$fg_made)
prop.table(table(pbp$fg_made))

head(pbp)

# Group weather & field situations into buckets and binarize
pbp <- pbp %>%
  mutate(rain = if_else(grepl("Rain", weather), 1, 0),
         snow = if_else(grepl("Snow", weather), 1, 0),
         precipitation = if_else(rain == 1 | snow == 1, 1, 0),
         freezing = if_else(!is.na(temp) & temp < 32, 1, 0),
         turf = if_else(grepl("turf", surface), 1, 0),
         grass = if_else(surface == 'grass', 1, 0),
         roof_closed = if_else(roof %in% c('outdoors','open'), 0, 1),
         high_altitude = if_else(grepl("mile high", stadium, ignore.case = TRUE), 1, 0)
  ) 

# Identify additional play/contextual information
pbp <- pbp %>%
  mutate(half_end = if_else((qtr == 2 & quarter_end == 1) | (qtr == 4 & quarter_end == 1), 1, 0),
         game_end = if_else((qtr == 4 & quarter_end == 1), 1, 0),
         post_season = if_else(season_type == 'POST', 1, 0),
         last_two_minutes = if_else((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120, 1, 0),
         total_points = posteam_score + defteam_score
  )

# Extract other situational factors
pbp <- pbp %>% 
  mutate(team_is_trailing = if_else(posteam_score < defteam_score, 1, 0),
         tie_game = if_else(posteam_score == defteam_score, 1, 0),
         timeouts_remaining = if_else(posteam == home_team, home_timeouts_remaining, away_timeouts_remaining),
         at_home = if_else(posteam == home_team, 1, 0),
         on_road = if_else(posteam == away_team, 1, 0)
  )

# Bucket game times into NFL slates
pbp <- pbp %>%
  mutate(
    # Convert to POSIXct datetime from string (in UTC), convert UTC to ET
    time_utc = ymd_hms(time_of_day, tz = "UTC"),
    time_et = with_tz(time_utc, tzone = "America/New_York"),
    hour_et = hour(time_et),
    # Assign slate based on ET hour
    slate = case_when(
      hour_et < 15 ~ "1pm Slate",
      hour_et >= 15 & hour_et < 20 ~ "4pm Slate",
      hour_et >= 20 ~ "8pm Slate",
      TRUE ~ "Other"),
    prime_time = if_else(slate == "8pm Slate", 1, 0)
  )

# Create features/indicators of special, high-pressure situations
pbp <- pbp %>%
  mutate(
    # FG to win the game
    kick_to_win = if_else(qtr == 4 & quarter_seconds_remaining < 120 & score_differential < 0 & abs(score_differential) <= 3, 1, 0),
    # FG to tie the game
    kick_to_tie = if_else(qtr == 4 & quarter_seconds_remaining < 120 & abs(score_differential) == 3, 1, 0),
    # Indicate long kicks (50+)
    long_kick = if_else(kick_distance >= 50, 1, 0)
  )

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

# Average FG Pct by season
pbp %>% group_by(season) %>% summarise(mean(fg_made))

# Plot average FG percent by season - Hypothesis: kickers have improved over the years
avg_fg_by_season <- pbp %>%
  group_by(season) %>%
  summarise(fg_pct = mean(fg_made, na.rm = TRUE)) %>%
  ggplot(aes(x = season, y = fg_pct)) +
  geom_line(color = "#0080C6", linewidth = 1) +
  geom_point(color = "#0080C6", size = 2) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
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

# FG% before 2005 = 80%, FG% since 2005 = 86%
pbp %>% 
  mutate(bucket = if_else(season < 2010, '2000-2009', '2010-2024')) %>% 
  group_by(bucket) %>%
  summarise(fg_pct = mean(fg_made, na.rm = TRUE)) 

# Drop all attempts before 2010 season - do not want to train on outdated information
pbp <- pbp %>% filter(season >= 2010)

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


