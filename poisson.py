'''
Module with functions to predict by Poisson probability distribution.
'''

import pickle
from operator import itemgetter

import pandas as pd
from scipy.stats import poisson

# Load training set teams statistics dictionary
with open("teams.pickle", "rb") as f:
    TEAMS = pickle.load(f)


def predict_game(home_team: str, away_team: str, teams: dict = TEAMS):
    ''' Predicts game outcome through Poisson.
    
    Inputs:
        - home_team, away_team: teams involved in game
        - teams: statistic dict with all teams in training set stats
        
    Outputs:
        - HS, AS: predicted game outcome as exact home and away team score
        - pr_home, pr_draw, pr_away: predicted WDL probabilities
    '''
    
    # Calculate the value of lambda (λ) for both Home Team and Away Team.
    ht_stats_dict = [t for t in teams if t["name"] == home_team][0]
    at_stats_dict = [t for t in teams if t["name"] == away_team][0]
    
    avg_ht_scored = ht_stats_dict["avg_goals_scored"]
    avg_ht_conceded = ht_stats_dict["avg_goals_against"]
    avg_at_scored = at_stats_dict["avg_goals_scored"]
    avg_at_conceded = at_stats_dict["avg_goals_against"]
    
    lambda_home_team = avg_ht_scored * avg_at_conceded
    lambda_away_team = avg_at_scored * avg_ht_conceded
    
    prd_W, prd_D, prd_L = 0, 0, 0
    result_probs = list()
    
    for x in range(0, 14):  # number of goals home team (max14)
        for y in range(0, 14):  # number of goals away team (max14)
            p = poisson.pmf(x, lambda_home_team) * poisson.pmf(y, lambda_away_team)
            result_probs.append((x, y, p))
            # if p > 0.01: print(f"Result: {x}-{y}, Prob: {p}")
            if x == y:
                prd_D += p
            elif x > y:
                prd_W += p
            else:
                prd_L += p
    
    HS, AS, P = max(result_probs, key=itemgetter(2))
      
    # print(f"{home_team}: {prd_W}")
    # print(f"Draw: {prd_D}")
    # print(f"{away_team}: {prd_L}")
    # print(f"Most probable result: {HS}-{AS} ({P})")
                 
    # print(lambda_home_team, lambda_away_team)
    
    return HS, AS, prd_W, prd_D, prd_L


def predict_dataset(df: pd.DataFrame, method: str):
    ''' Predicts each game in DF and writes predicted
    outcomes for all games in dataset (game outcome as
    HS/AS and prd_WDL probabilities).
    
    Inputs:
        - df: pd.DF dataset with games to be predicted
        
    Output:
    
    '''
    
    if ['HT', 'AT'] not in df.columns:
        raise ValueError("DF must contain HT, AT columns.")
    
    df_with_preds = df.copy()
    
    for i, game in df_with_preds.iterrows():
        
        # Extract HT and AT to predict game outcome
        HT = game.HT
        AT = game.AT
        # Ignore games with teams not present in training set
        try:
            HS, AS, W, D, L = predict_game(HT, AT)
        except:
            continue
        
        # Fill in predicted outcomes of the game
        df_with_preds.at[i, "pr_HS"] = HS
        df_with_preds.at[i, "prd_AS"] = AS
        
        df_with_preds.at[i, "prd_W"] = W
        df_with_preds.at[i, "prd_D"] = D
        df_with_preds.at[i, "prd_L"] = L
        
    return df_with_preds
