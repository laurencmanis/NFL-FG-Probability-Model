############################################# NFL FIELD GOAL PROBABILITY MODEL ############################################# 
################################### 4.0) FIELD GOALS ABOVE EXPECTED & KICKER RANKINGS ######################################

source('03_Modeling.R')

head(all_data)

#----------------------------------------------------- FGAx -------------------------------------------------------#

# Compute field goals above expected for each kicker in 2024, similar to NHL goals saved above expected stat (GSAx)
fgax <- all_data %>%
  filter(season == 2024) %>%
  group_by(kicker_player_id, kicker_player_name, posteam) %>%
  summarise(
    FG_attempted = sum(field_goal_attempt),
    FG_made = sum(fg_made),
    FG_pct = FG_made / FG_attempted,
    xFG = sum(fg_probability), 
    FGAx = FG_made - xFG,
    xFG_pct = xFG / FG_attempted,
    FGpct_Ax = FG_pct - xFG_pct
  ) %>%
  # Eliminate kickers with fewer than 10 attempts in the season
  filter(FG_attempted >= 10) %>%
  arrange(desc(FGAx))

head(fgax, 10)

# Field Goals Above Expected (FGAx) 
# Metric inspired by hockey's Goals Saved Above Expected, (GSAx), commonly used to evaluate goaltenders. 
# Rather than simply counting total field goals made (FGM) or raw field goal percentage (FG%), 
# FGAx compares each kicker's actual makes to what would be expected based on kick difficulty, including factors like distance, weather, game situation, and more.
# By accounting for the quality and context of each attempt, FGAx provides a more accurate and fair assessment of performance. 
# It rewards kickers who convert difficult kicks and adjusts for those benefiting from favorable conditions or only taking short, easier attempts. 
# This helps identify true overperformers, not just those with a high volume of short-range kicks.
