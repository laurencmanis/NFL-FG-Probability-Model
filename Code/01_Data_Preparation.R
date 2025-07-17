############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
################################################## 1.0) DATA PREPARATION ################################################### 
library(dplyr)
library(tidyverse)
library(ggplot2)
library(nflfastR)
library(stringr)
library(glmnet)
library(caret)
library(lubridate)
library(leaps)
library(ggtext)
library(pROC)
library(randomForest)
library(car)
library(yardstick)

#---------------------------------------------------- DATA COLLECTION ----------------------------------------------------#

# Load in play by play data for all games since the 2000 season
pbp <- load_pbp(2000:2024)

# Select columns relevant to field goals
pbp <- pbp %>% 
  select(
  # General Game Information 
  season, season_type, week, game_date, game_id, away_team, home_team, spread_line, total_line, 
  # Environmental Features
  time_of_day, location, stadium, weather, roof, surface, temp, wind, 
  # Play-Level Information
  play_id, posteam, defteam, posteam_type, game_half, qtr, play_clock, quarter_seconds_remaining, 
  quarter_end, time, drive, down, side_of_field, yardline_100, yrdln, ydstogo, play_type, desc, series_result,
  # Situational Features
  score_differential, posteam_score, defteam_score, score_differential_post, ep, epa, wp, wpa, 
  home_timeouts_remaining, away_timeouts_remaining, timeout, timeout_team, 
  # Field Goal Information
  field_goal_attempt, field_goal_result, kick_distance, kicker_player_name, kicker_player_id
  )


#---------------------------------------------------- DATA VALIDATION ----------------------------------------------------#

# Ensure correct number of games per season (~272 regular season games)
pbp %>% group_by(season) %>% summarise(n_games = n_distinct(game_id))

# Ensure a reasonable number of plays per game - no games with fewer than 139 plays, no games with more than 245 plays
pbp %>% group_by(game_id) %>% summarise(n_plays = n_distinct(play_id)) %>% arrange(-n_plays)

# Ensure a credible number of field goal attempts per season - ~910-1000
pbp %>% filter(field_goal_attempt == 1) %>% group_by(season) %>% summarise(fga = n_distinct(play_id)) 

# Ensure all teams are present each season
pbp %>% group_by(season) %>% summarise(teams = n_distinct(home_team))

# Verify that play ids are unique within games
pbp %>% group_by(game_id, play_id) %>% tally() %>% filter(n > 1)

# Look at missing values by column 
sapply(pbp, function(x) sum(is.na(x)))

# Missing some temp/wind values - can instead extract these from weather column, which has fewer null values
pbp <- pbp %>%
  mutate(
    temp = str_extract(weather, "Temp: \\d+") %>% str_extract("\\d+") %>% as.numeric(),
    wind = str_extract(weather, "Wind: [A-Za-z ]*\\d+") %>% str_extract("\\d+") %>% as.numeric(),
    humidity = str_extract(weather, "Humidity: \\d+") %>% str_extract("\\d+") %>% as.numeric()
  ) 

# If temp/wind/humidity is NA, replace with corresponding means where reasonable (ie, roof closed situations), drop where we cannot impute 
means_roof_closed <- pbp %>% 
  ungroup() %>%
  filter(roof == 'closed') %>% 
  summarise(
    mean_temp = mean(temp, na.rm = TRUE),
    mean_wind = mean(wind, na.rm = TRUE),
    mean_humidity = mean(humidity, na.rm = TRUE)
  )

overall_mean_humidity <- mean(pbp$humidity, na.rm = TRUE)

pbp <- pbp %>%
  mutate(
    temp = if_else(is.na(temp) & roof == 'closed', means_roof_closed$mean_temp, temp),
    wind = if_else(is.na(wind) & roof == 'closed', means_roof_closed$mean_wind, wind),
    humidity = if_else(is.na(humidity), if_else(
      roof == 'closed', means_roof_closed$mean_humidity, overall_mean_humidity), humidity)
  ) %>%
  filter(!is.na(wind) & !is.na(temp))

# Other NAs (not significant for analysis)
# - Missing `posteam` / `defteam`: mostly non-plays like quarter ends, timeouts, etc.
# - Missing `down`: typically kickoffs, extra points, or other special teams plays
# - Missing `yrdln` / `yardline_100`: often non-plays such as timeouts or game end
# These will be irrelevant for field goal modeling and can be ignored or dropped later

# Identify additional game context/situations, before dropping non-field goal rows 
pbp <- pbp %>%
  arrange(game_id, qtr, desc(quarter_seconds_remaining), play_id) %>% 
  group_by(game_id) %>%
  mutate(prior_play_type = lag(play_type),
         timeout_prior = lag(timeout)
  ) 

# Drop all rows not containing a field goal attempt
pbp <- pbp %>% filter(!is.na(field_goal_attempt) & field_goal_attempt == 1 & play_type == 'field_goal') %>% distinct()

# Check null values by column again
sapply(pbp, function(x) sum(is.na(x)))

# Create unique id for each observation
pbp$fg_id <- paste0(pbp$game_id, pbp$play_id)

# Ensure only one row/play per field goal attempt 
pbp %>% group_by(fg_id) %>% summarise(n_rows = n()) %>% arrange(-n_rows)

# Ensure all rows are actual field goal attempts 
unique(pbp$play_type)
unique(pbp$field_goal_attempt)

# Ensure no rows are missing any key information (kick distance, wind, field surface, etc.) - 
sum(is.na(pbp$wind))
sum(is.na(pbp$kick_distance))
sum(is.na(pbp$surface))

# Total number of observations - ~15K
n_distinct(pbp$fg_id)
sum(pbp$field_goal_attempt)

# Distribution of field goal results 
table(pbp$field_goal_result)
prop.table(table(pbp$field_goal_result))

# Drop blocked field goals from the data set - Only ~2% of observations are blocked, do not want to train on this
# Also would not want to base a coaching decision on this 
pbp <- pbp %>%
  filter(field_goal_result != 'blocked')

head(pbp)
