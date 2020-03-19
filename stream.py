import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

df_scaled2 = pd.read_csv('df_scaled2.csv', index_col='Team')
df_scaled2.head()
team_list = df_scaled2.index.tolist()
# team_list

df_stats_poss3 = pd.read_csv('df_stats_poss3.csv', index_col='Team')
df_stats_poss3.head()
cos_sim1 = cosine_similarity(df_scaled2)
# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
df_leagues = pd.read_csv('df_stats_leagues.csv', index_col=0)
indices = pd.Series(df_stats_poss3.index)
#  defining the function that takes in movie title
# as input and returns the top 10 recommended movies
def recommendations3(team, cosine_sim = cos_sim1):

    epl_array = (df_leagues['team'][df_leagues['league'] == 'EPL']).tolist()
    ligue_1_array = (df_leagues['team'][df_leagues['league'] == 'Ligue_1']).tolist()
    serie_a_array = (df_leagues['team'][df_leagues['league'] == 'Serie_A']).tolist()
    la_liga_array = (df_leagues['team'][df_leagues['league'] == 'La_liga']).tolist()
    bundesliga_array = (df_leagues['team'][df_leagues['league'] == 'Bundesliga']).tolist()

    en_teams = []
    fr_teams = []
    it_teams = []
    sp_teams = []
    ge_teams = []
    to_play = []

    # initializing the empty list of recommended movies
    recommended_teams = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == team].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    # getting the indexes of the 10 most similar movies
    all_indexes = list(score_series.index)

    # populating the list with the titles of the best 10 matching movies
    for i in all_indexes:
        recommended_teams.append(list(df_stats_poss3.index)[i])

    if recommended_teams[0] in epl_array:
        for i in recommended_teams:
            if i in ligue_1_array:
                fr_teams.append(i)
            if i in serie_a_array:
                it_teams.append(i)
            if i in la_liga_array:
                sp_teams.append(i)
            if i in bundesliga_array:
                ge_teams.append(i)
        to_play.append(fr_teams[0])
        to_play.append(it_teams[0])
        to_play.append(sp_teams[0])
        to_play.append(ge_teams[0])


    elif recommended_teams[0] in ligue_1_array:
        for i in recommended_teams:
            if i in epl_array:
                en_teams.append(i)
            if i in serie_a_array:
                it_teams.append(i)
            if i in la_liga_array:
                sp_teams.append(i)
            if i in bundesliga_array:
                ge_teams.append(i)
        to_play.append(en_teams[0])
        to_play.append(it_teams[0])
        to_play.append(sp_teams[0])
        to_play.append(ge_teams[0])

    elif recommended_teams[0] in serie_a_array:
        for i in recommended_teams:
            if i in epl_array:
                en_teams.append(i)
            if i in ligue_1_array:
                fr_teams.append(i)
            if i in la_liga_array:
                sp_teams.append(i)
            if i in bundesliga_array:
                ge_teams.append(i)
        to_play.append(en_teams[0])
        to_play.append(fr_teams[0])
        to_play.append(sp_teams[0])
        to_play.append(ge_teams[0])

    elif recommended_teams[0] in la_liga_array:
        for i in recommended_teams:
            if i in epl_array:
                en_teams.append(i)
            if i in serie_a_array:
                it_teams.append(i)
            if i in ligue_1_array:
                fr_teams.append(i)
            if i in bundesliga_array:
                ge_teams.append(i)
        to_play.append(en_teams[0])
        to_play.append(fr_teams[0])
        to_play.append(it_teams[0])
        to_play.append(ge_teams[0])

    elif recommended_teams[0] in bundesliga_array:
        for i in recommended_teams:
            if i in epl_array:
                en_teams.append(i)
            if i in serie_a_array:
                it_teams.append(i)
            if i in la_liga_array:
                sp_teams.append(i)
            if i in ligue_1_array:
                fr_teams.append(i)
        to_play.append(en_teams[0])
        to_play.append(fr_teams[0])
        to_play.append(it_teams[0])
        to_play.append(sp_teams[0])

    else:
        print('not applicable')

    return to_play

df_merged2 = pd.read_csv('df_merged2.csv')
df_merged2 = df_merged2.drop(df_merged2.columns[0], axis=1)
df_merged2.head()

col_list1 = df_merged2.columns[62:91]
col_list2 = df_merged2.columns[6:62]
col_list2 = col_list2.append(df_merged2.columns[1:3])

#df_merged2['score'] = df_merged2['home_goals'] - df_merged2['away_goals']
results = df_merged2.drop(['home_goals', 'away_goals', 'Home', 'Away'], 1)
results['winner'] = None
results['winner'][results.score > 0] = 1
results['winner'][results.score < 0] = 2
results['winner'][results.score == 0] = 0
#results = results[results.winner != 0]

y = results['winner'].astype(int)
X = results[col_list1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

lr = LogisticRegression(C=0.1)
lr.fit(X_train, y_train)
training_preds2 = lr.predict(X_train)
test_preds2 = lr.predict(X_test)
training_accuracy2 = accuracy_score(y_train, training_preds2)
test_accuracy2 = accuracy_score(y_test, test_preds2)
lr_f1 = f1_score(y_test, test_preds2, average='macro')
lr_p = precision_score(y_test, test_preds2, average='macro')
lr_r = recall_score(y_test, test_preds2, average='macro')

# print('F1 score: {:.4}%'.format(lr_f1 * 100))
# print('precision score: {:.4}%'.format(lr_p * 100))
# print('recall score: {:.4}%'.format(lr_r * 100))




def match3(df, team1, team2, model):

    match = pd.DataFrame(columns=col_list2, index=[0])

    match['ppda_coef_x'] = np.random.normal(df[df.Home == team1]['ppda_coef_x'].iloc[0], scale = df['ppda_coef_x'].std())
    match['deep_x'] = np.random.normal(df[df.Home == team1]['deep_x'].iloc[0], scale = df['deep_x'].std())
    match['CR_x'] = np.random.normal(df[df.Home == team1]['CR_x'].iloc[0], scale = df['CR_x'].std())
    match['P_x'] = np.random.normal(df[df.Home == team1]['P_x'].iloc[0], scale = df['P_x'].std())
    match['DR_x'] = np.random.normal(df[df.Home == team1]['DR_x'].iloc[0], scale = df['DR_x'].std())
    match['IBS_x'] = np.random.normal(df[df.Home == team1]['IBS_x'].iloc[0], scale = df['IBS_x'].std())
    match['OBS_x'] = np.random.normal(df[df.Home == team1]['OBS_x'].iloc[0], scale = df['OBS_x'].std())
    match['TBOX_x'] = np.random.normal(df[df.Home == team1]['TBOX_x'].iloc[0], scale = df['TBOX_x'].std())
    match['FKCR_x'] = np.random.normal(df[df.Home == team1]['FKCR_x'].iloc[0], scale = df['FKCR_x']. std())
    match['CRN_x'] = np.random.normal(df[df.Home == team1]['CRN_x'].iloc[0], scale = df['CRN_x'].std())
    match['CRNCR_x'] = np.random.normal(df[df.Home == team1]['CRNCR_x'].iloc[0], scale = df['CRNCR_x'].std())
    match['FKS_x'] = np.random.normal(df[df.Home == team1]['FKS_x'].iloc[0], scale = df['FKS_x'].std())
    match['TOFF_x'] = np.random.normal(df[df.Home == team1]['TOFF_x'].iloc[0], scale = df['TOFF_x'].std())
    match['BCS_x'] = np.random.normal(df[df.Home == team1]['BCS_x'].iloc[0], scale = df['BCS_x'].std())
    match['oppda_coef_x'] = np.random.normal(df[df.Home == team1]['oppda_coef_x'].iloc[0], scale = df['oppda_coef_x'].std())
    match['deep_allowed_x'] = np.random.normal(df[df.Home == team1]['deep_allowed_x'].iloc[0], scale = df['deep_allowed_x'].std())
    match['INT_x'] = np.random.normal(df[df.Home == team1]['INT_x'].iloc[0], scale = df['INT_x'].std())
    match['BLK_x'] = np.random.normal(df[df.Home == team1]['BLK_x'].iloc[0], scale = df['BLK_x'].std())
    match['TKL_x'] = np.random.normal(df[df.Home == team1]['TKL_x'].iloc[0], scale = df['TKL_x'].std())
    match['FC_x'] = np.random.normal(df[df.Home == team1]['FC_x'].iloc[0], scale = df['FC_x'].std())
    match['AW_x'] = np.random.normal(df[df.Home == team1]['AW_x'].iloc[0], scale = df['AW_x'].std())
    match['BR_x'] = np.random.normal(df[df.Home == team1]['BR_x'].iloc[0], scale = df['BR_x'].std())
    match['DW_x'] = np.random.normal(df[df.Home == team1]['DW_x'].iloc[0], scale = df['DW_x'].std())
    match['AKS_x'] = np.random.normal(df[df.Home == team1]['AKS_x'].iloc[0], scale = df['AKS_x'].std())
    match['CL_x'] = np.random.normal(df[df.Home == team1]['CL_x'].iloc[0], scale = df['CL_x'].std())
    match['PUNCH_x'] = np.random.normal(df[df.Home == team1]['PUNCH_x'].iloc[0], scale = df['PUNCH_x'].std())
    match['LMT_x'] = np.random.normal(df[df.Home == team1]['LMT_x'].iloc[0], scale = df['LMT_x'].std())
    match['poss_x'] = np.random.normal(df[df.Home == team1]['poss_x'].iloc[0], scale = df['poss_x'].std())
    match['xG'] = np.random.normal((df[df.Home == team1]['xG'].mean() + df[df.Away == team1]['xG.1'].mean())/2, scale = (df['xG'].std() + df['xG.1'].std())/2)

    match['ppda_coef_y'] = np.random.normal(df[df.Away == team2]['ppda_coef_y'].iloc[0], scale = df['ppda_coef_y'].std())
    match['deep_y'] = np.random.normal(df[df.Away == team2]['deep_y'].iloc[0], scale= df['deep_y'].std())
    match['CR_y'] = np.random.normal(df[df.Away == team2]['CR_y'].iloc[0], scale= df['CR_y'].std())
    match['P_y'] = np.random.normal(df[df.Away == team2]['P_y'].iloc[0], scale= df['P_y'].std())
    match['DR_y'] = np.random.normal(df[df.Away == team2]['DR_y'].iloc[0], scale= df['DR_y'].std())
    match['IBS_y'] = np.random.normal(df[df.Away == team2]['IBS_y'].iloc[0], scale= df['IBS_y'].std())
    match['OBS_y'] = np.random.normal(df[df.Away == team2]['OBS_y'].iloc[0], scale= df['OBS_y'].std())
    match['TBOX_y'] = np.random.normal(df[df.Away == team2]['TBOX_y'].iloc[0], scale= df['TBOX_y'].std())
    match['FKCR_y'] = np.random.normal(df[df.Away == team2]['FKCR_y'].iloc[0], scale= df['FKCR_y'].std())
    match['CRN_y'] = np.random.normal(df[df.Away == team2]['CRN_y'].iloc[0], scale= df['CRN_y'].std())
    match['CRNCR_y'] = np.random.normal(df[df.Away == team2]['CRNCR_y'].iloc[0], scale= df['CRNCR_y'].std())
    match['FKS_y'] = np.random.normal(df[df.Away == team2]['FKS_y'].iloc[0], scale= df['FKS_y'].std())
    match['TOFF_y'] = np.random.normal(df[df.Away == team2]['TOFF_y'].iloc[0], scale= df['TOFF_y'].std())
    match['BCS_y'] = np.random.normal(df[df.Away == team2]['BCS_y'].iloc[0], scale= df['BCS_y'].std())
    match['oppda_coef_y'] = np.random.normal(df[df.Away == team2]['oppda_coef_y'].iloc[0], scale= df['oppda_coef_y'].std())
    match['deep_allowed_y'] = np.random.normal(df[df.Away == team2]['deep_allowed_y'].iloc[0], scale= df['deep_allowed_y'].std())
    match['INT_y'] = np.random.normal(df[df.Away == team2]['INT_y'].iloc[0], scale= df['INT_y'].std())
    match['BLK_y'] = np.random.normal(df[df.Away == team2]['BLK_y'].iloc[0], scale= df['BLK_y'].std())
    match['TKL_y'] = np.random.normal(df[df.Away == team2]['TKL_y'].iloc[0], scale= df['TKL_y'].std())
    match['FC_y'] = np.random.normal(df[df.Away == team2]['FC_x'].iloc[0], scale= df['FC_y'].std())
    match['AW_y'] = np.random.normal(df[df.Away == team2]['AW_y'].iloc[0], scale= df['AW_y'].std())
    match['BR_y'] = np.random.normal(df[df.Away == team2]['BR_y'].iloc[0], scale= df['BR_y'].std())
    match['DW_y'] = np.random.normal(df[df.Away == team2]['DW_y'].iloc[0], scale= df['DW_y'].std())
    match['AKS_y'] = np.random.normal(df[df.Away == team2]['AKS_y'].iloc[0], scale= df['AKS_y'].std())
    match['CL_y'] = np.random.normal(df[df.Away == team2]['CL_y'].iloc[0], scale= df['CL_y'].std())
    match['PUNCH_y'] = np.random.normal(df[df.Away == team2]['PUNCH_y'].iloc[0], scale= df['PUNCH_y'].std())
    match['LMT_y'] = np.random.normal(df[df.Away == team2]['LMT_y'].iloc[0], scale= df['LMT_y'].std())
    match['poss_y'] = np.random.normal(df[df.Away == team2]['poss_y'].iloc[0], scale= df['poss_y'].std())
    match['xG.1'] = np.random.normal((df[df.Home == team2]['xG'].mean() + df[df.Away == team2]['xG.1'].mean())/2, scale= (df['xG'].std() + df['xG.1'].std())/2)

    match['xG_diff'] = match['xG'] - match['xG.1']
    match['ppda_diff'] = match['ppda_coef_x'] - match['ppda_coef_y']
    match['deep_diff'] = match['deep_x'] - match['deep_y']
    match['CR_diff'] = match['CR_x'] - match['CR_y']
    match['P_diff'] = match['P_x'] - match['P_y']
    match['DR_diff'] = match['DR_x'] - match['DR_y']
    match['IBS_diff'] = match['IBS_x'] - match['IBS_y']
    match['OBS_diff'] = match['OBS_x'] - match['OBS_y']
    match['TBOX_diff'] = match['TBOX_x'] - match['TBOX_y']
    match['FKCR_diff'] = match['FKCR_x'] - match['FKCR_y']
    match['CRN_diff'] = match['CRN_x'] - match['CRN_y']
    match['CRNCR_diff'] = match['CRNCR_x'] - match['CRNCR_y']
    match['FKS_diff'] = match['FKS_x'] - match['FKS_y']
    match['TOFF_diff'] = match['TOFF_x'] - match['TOFF_y']
    match['BCS_diff'] = match['BCS_x'] - match['BCS_y']
    match['oppda_diff'] = match['oppda_coef_x'] - match['oppda_coef_y']
    match['deep_allowed_diff'] = match['deep_allowed_x'] - match['deep_allowed_y']
    match['INT_diff'] = match['INT_x'] - match['INT_y']
    match['BLK_diff'] = match['BLK_x'] - match['BLK_y']
    match['TKL_diff'] = match['TKL_x'] - match['TKL_y']
    match['FC_diff'] = match['FC_x'] - match['FC_y']
    match['AW_diff'] = match['AW_x'] - match['AW_y']
    match['BR_diff'] = match['BR_x'] - match['BR_y']
    match['DW_diff'] = match['DW_x'] - match['DW_y']
    match['AKS_diff'] = match['AKS_x'] - match['AKS_y']
    match['CL_diff'] = match['CL_x'] - match['CL_y']
    match['PUNCH_diff'] = match['PUNCH_x'] - match['PUNCH_y']
    match['LMT_diff'] = match['LMT_x'] - match['LMT_y']
    match['poss_diff'] = match['poss_x'] - match['poss_y']

    match = match[col_list1]

    match_array = match.values

    prediction = model.predict(match_array)

    winner = None

    if prediction == 1:
        winner = team1
    elif prediction == 2:
        winner = team2
    else:
        winner = 'Tie'

    return winner



def simulate_matches2(team, team2, n_matches=50):
    match_results = []
    for i in range(n_matches):
        match_results.append(match3(df_merged2, team, team2, lr))
    team1_proba = match_results.count(team)/len(match_results)*100
    team2_proba = match_results.count(team2)/len(match_results)*100

    print(team, str(round(team1_proba, 2)) + '%')
    print(team2, str(round(team2_proba,2)) + '%')
    print('-------------------------')
    print()

    if team1_proba > team2_proba:
        overall_winner = team
    elif team2_proba > team1_proba:
        overall_winner = team2
    else:
        overall_winner = 'Tie'

    return
def rec_sim(team):
    teams_to_play = recommendations3(team)
    for team2 in teams_to_play:
        matchup = simulate_matches2(team, team2, n_matches=25)
    return matchup















team_list

rec_sim('Southampton')
