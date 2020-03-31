# Soccer Recommender and Simulator

slideshow - https://docs.google.com/presentation/d/1ZF9b5ImsS19IaxzGQsibQVaTjdoJkYbP_f699EQW1us/edit?usp=sharing

# Project Goals

  Soccer is the most popular sport in the world, and European soccer has the most prestigious teams in the world. But many fans focus their attention on one league and have a favorite team in that league. This project aims to broaden horizons and suggest teams in the other European Leagues.
  
  Take a team from one of the top five European Leagues -- English Premier League, French Ligue 1, Italian Serie A, German Bundesliga, and Spanish La Liga -- and return the most similar team from each of the other four teams.
  
  Then simulate a series of games in each matchup to see how your team would fare against the other four
  
# Data

  - Advanced analytics dataset for 2018-2019 season from Kaggle
  - Other stats for 2018-2019 season from Rotowire
  - 2018-2019 season game results from FBref.com
  
# Recommendation System

  First manipulated statistic by converting all stats to per game averages, then further converting offensive stats proportionate to how often that team had the ball, and defensive stats proportionate to how often that team did not have the ball.
  
  Used cosine similarity to get the most similar team from each one of the other four leagues
  
# Simulator
  
  Final Model used logistic regression
    
    Training accuracy: 59.28%
    Validation accuracy: 56.14%
    F1 Score: 47.53%
    Precision: 48.98%
    Recall: 49.9%
  
 
