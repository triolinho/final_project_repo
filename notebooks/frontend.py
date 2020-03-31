import streamlit as st
from stream import rec_sim, simulate_matches2, recommendations3, match3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import *


df_scaled2 = pd.read_csv('df_scaled2.csv', index_col='Team')
team_list = df_scaled2.index.tolist()
df_stats_poss3 = pd.read_csv('df_stats_poss3.csv', index_col='Team')
cos_sim1 = cosine_similarity(df_scaled2)
df_leagues = pd.read_csv('df_stats_leagues.csv', index_col=0)
indices = pd.Series(df_stats_poss3.index)
df_merged2 = pd.read_csv('df_merged2.csv')
df_merged2 = df_merged2.drop(df_merged2.columns[0], axis=1)
col_list1 = df_merged2.columns[62:91]
col_list2 = df_merged2.columns[6:62]
col_list2 = col_list2.append(df_merged2.columns[1:3])
results = df_merged2.drop(['home_goals', 'away_goals', 'Home', 'Away'], 1)
results['winner'] = None
results['winner'][results.score > 0] = 1
results['winner'][results.score < 0] = 2
results['winner'][results.score == 0] = 0
y = results['winner'].astype(int)
X = results[col_list1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
lr = LogisticRegression(C=0.1)
lr.fit(X_train, y_train)

def simulate_matches2(team, team2, n_matches=50):
    match_results = []
    for i in range(n_matches):
        match_results.append(match3(df_merged2, team, team2, lr))
    team1_proba = match_results.count(team)/len(match_results)*100
    team2_proba = match_results.count(team2)/len(match_results)*100

    st.write(team, str(round(team1_proba, 2)) + '%')
    st.write(team2, str(round(team2_proba,2)) + '%')
    st.write('-------------------------')
    st.write()

    if team1_proba > team2_proba:
        overall_winner = team
    elif team2_proba > team1_proba:
        overall_winner = team2
    else:
        overall_winner = 'Tie'

    return

team = st.selectbox('What is your favorite team?', team_list)

if team != "":

    rec_sim(team)
