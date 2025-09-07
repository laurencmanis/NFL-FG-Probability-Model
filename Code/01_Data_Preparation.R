############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
################################################## 1.0) DATA PREPARATION ################################################### 

source('00_Libraries_Functions.R')

#---------------------------------------------------- DATA COLLECTION ----------------------------------------------------#

# Load in play by play data for all games since the 2000 season
data <- load_pbp(2000:2024)

# Select columns relevant to field goals
data <- data %>% 
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
data %>% group_by(season) %>% summarise(n_games = n_distinct(game_id))

# Ensure a reasonable number of plays per game - no games with fewer than 139 plays, no games with more than 245 plays
data %>% group_by(game_id) %>% summarise(n_plays = n_distinct(play_id)) %>% arrange(-n_plays)

# Ensure a credible number of field goal attempts per season - ~910-1000
data %>% filter(field_goal_attempt == 1) %>% group_by(season) %>% summarise(fga = n_distinct(play_id)) 

# Ensure all teams are present each season
data %>% group_by(season) %>% summarise(teams = n_distinct(home_team))

# Verify that play ids are unique within games
data %>% group_by(game_id, play_id) %>% tally() %>% filter(n > 1)

# Look at missing values by column 
sapply(data, function(x) sum(is.na(x)))

# Missing some temp/wind values - can instead extract these from weather column, which has fewer null values
data <- data %>%
  mutate(
    temp = str_extract(weather, "Temp: \\d+") %>% str_extract("\\d+") %>% as.numeric(),
    wind = str_extract(weather, "Wind: [A-Za-z ]*\\d+") %>% str_extract("\\d+") %>% as.numeric(),
    humidity = str_extract(weather, "Humidity: \\d+") %>% str_extract("\\d+") %>% as.numeric()
  ) 

# If temp/wind/humidity is NA, replace with corresponding means where reasonable (ie, roof closed situations), drop where we cannot impute 
means_roof_closed <- data %>% 
  ungroup() %>%
  filter(roof == 'closed') %>% 
  summarise(
    mean_temp = mean(temp, na.rm = TRUE),
    mean_wind = mean(wind, na.rm = TRUE),
    mean_humidity = mean(humidity, na.rm = TRUE)
  )

overall_mean_humidity <- mean(data$humidity, na.rm = TRUE)

data <- data %>%
  mutate(
    temp = if_else(is.na(temp) & roof == 'closed', means_roof_closed$mean_temp, temp),
    wind = if_else(is.na(wind) & roof == 'closed', means_roof_closed$mean_wind, wind),
    humidity = if_else(is.na(humidity), if_else(
      roof == 'closed', means_roof_closed$mean_humidity, overall_mean_humidity), humidity)
  ) %>%
  filter(!is.na(wind) & !is.na(temp))

# Group weather & field situations into buckets and binarize
data <- data %>%
  mutate(rain = if_else(grepl("Rain", weather), 1, 0),
         snow = if_else(grepl("Snow", weather), 1, 0),
         precipitation = if_else(rain == 1 | snow == 1, 1, 0),
         freezing = if_else(!is.na(temp) & temp < 32, 1, 0),
         turf = if_else(grepl("turf", surface), 1, 0),
         grass = if_else(surface == 'grass', 1, 0),
         roof_closed = if_else(roof %in% c('outdoors','open'), 0, 1),
         high_altitude = if_else(grepl("mile high", stadium, ignore.case = TRUE), 1, 0)
  ) 

# Other NAs (not significant for analysis)
# - Missing `posteam` / `defteam`: mostly non-plays like quarter ends, timeouts, etc.
# - Missing `down`: typically kickoffs, extra points, or other special teams plays
# - Missing `yrdln` / `yardline_100`: often non-plays such as timeouts or game end
# These will be irrelevant for field goal modeling and can be ignored or dropped later

# Identify additional game context/situations, before dropping non-field goal rows 
data <- data %>%
  arrange(game_id, qtr, desc(quarter_seconds_remaining), play_id) %>% 
  group_by(game_id) %>%
  mutate(prior_play_type = lag(play_type),
         timeout_prior = lag(timeout)
  ) 

# Ensure no rows are still missing any key information  
sum(is.na(data$wind))
sum(is.na(data$surface))

# Identify additional play/contextual information
data <- data %>%
  mutate(half_end = if_else((qtr == 2 & quarter_end == 1) | (qtr == 4 & quarter_end == 1), 1, 0),
         game_end = if_else((qtr == 4 & quarter_end == 1), 1, 0),
         post_season = if_else(season_type == 'POST', 1, 0),
         last_two_minutes = if_else((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120, 1, 0),
         total_points = posteam_score + defteam_score
  )

# Extract other situational factors
data <- data %>% 
  mutate(team_is_trailing = if_else(posteam_score < defteam_score, 1, 0),
         tie_game = if_else(posteam_score == defteam_score, 1, 0),
         timeouts_remaining = if_else(posteam == home_team, home_timeouts_remaining, away_timeouts_remaining),
         at_home = if_else(posteam == home_team, 1, 0),
         on_road = if_else(posteam == away_team, 1, 0)
  )

# Bucket game times into NFL slates
data <- data %>%
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
data <- data %>%
  mutate(
    # FG to win the game
    kick_to_win = if_else(play_type == 'field_goal' & 
                            qtr == 4 & quarter_seconds_remaining < 120 & 
                            score_differential < 0 & abs(score_differential) <= 3, 1, 0),
    # FG to tie the game
    kick_to_tie = if_else(play_type == 'field_goal' & 
                            qtr == 4 & quarter_seconds_remaining < 120 & 
                            abs(score_differential) == 3, 1, 0),
    # Indicate long kicks (50+)
    long_kick = if_else(kick_distance >= 50, 1, 0)
  )

# Drop all rows not containing a field goal attempt
pbp <- data %>% filter(!is.na(field_goal_attempt) & field_goal_attempt == 1 & play_type == 'field_goal') %>% distinct()

# Check null values by column again
sapply(pbp, function(x) sum(is.na(x)))
sum(is.na(pbp$kick_distance))

# Create unique id for each observation
pbp$fg_id <- paste0(pbp$game_id, pbp$play_id)

# Ensure only one row/play per field goal attempt 
pbp %>% group_by(fg_id) %>% summarise(n_rows = n()) %>% arrange(-n_rows)

# Ensure all rows are actual field goal attempts 
unique(pbp$play_type)
unique(pbp$field_goal_attempt)

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

#---------------------------------------------------- DATA PREPARATION ----------------------------------------------------#

# Binarize Field Goal Information & Create Target Variable
pbp <- pbp %>%
  mutate(fg_made = if_else(field_goal_result == 'made', 1, 0),
         fg_missed = if_else(field_goal_result != 'made', 1, 0)) %>%
  ungroup()

# Total number of field goals made 
sum(pbp$fg_made)

# Proportion of field goals made 
mean(pbp$fg_made)
table(pbp$fg_made)
prop.table(table(pbp$fg_made))

head(pbp)

