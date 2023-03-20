'''
Module with functions to evaluate predictions
'''


import numpy as np
import pandas as pd


def evaluate_validation(preds_df: pd.DataFrame):
    '''
    Evaluation of predictions in validation context.
    
    Args:
        - preds_df: pd.DF with actual and prediction columns
    
    Output:
        - RMSE_avg: averaged RMSE of all games in set
        - RPS_avg: averaged RPS of all games in set
    '''
    
    # Actual and predicted columns
    actual_cols = ['HT', 'AT', 'HS', 'AS', 'GD', 'WDL']
    pred_cols = ['pr_HS', 'prd_AS', 'prd_W', 'prd_D', 'prd_L']
    
    for col in (actual_cols + pred_cols):
        if col not in list(preds_df.columns):
            raise ValueError(f"DF must contain {actual_cols + pred_cols}")
    
    for i, game in preds_df.iterrows():
        # Evaluate predicted exact score with RMSE
        rmse = (game.HS - game.pr_HS)**2 + (game.AS - game.prd_AS)**2
        preds_df.at[i, "RMSE"] = rmse
        
        # Evaluate predicted WDL probabilities with RPS
        if game.WDL == "W":
            actual_probs = [1, 0, 0]
        elif game.WDL == "D":
            actual_probs = [0, 1, 0]
        elif game.WDL == "L":
            actual_probs = [0, 0, 1]
        pred_probs = [game.prd_W, game.prd_D, game.prd_L]
        rps_v = rps_func(pred_probs, actual_probs)
        preds_df.at[i, "RPS"] = rps_v
        
        # acc, probs = 0, 0 # out of loop
        # if game.HS == int(game.pr_HS) and game.AS == int(game.prd_AS):
        #     acc += 1
        #     #print([game])
        # condition1 = (game.WDL == "W" and game.prd_W > game.prd_D and game.prd_W > game.prd_L)
        # condition2 = (game.WDL == "D" and game.prd_D > game.prd_W and game.prd_D > game.prd_L)
        # condition3 = (game.WDL == "L" and game.prd_L > game.prd_W and game.prd_L > game.prd_D)
        # if condition1 or condition2 or condition3:
        #     probs += 1
        # print(acc, probs) #out of loop

    n_games = preds_df.shape[0]
    RMSE_avg = np.sqrt(sum(preds_df["RMSE"].values) / n_games)
    RPS_avg = sum(preds_df["RPS"].values) / n_games
    
    return RMSE_avg, RPS_avg


def rps_func(pred_probs: list, actual_probs: list):
    ''' Computes Ranked Probability Score (RPS).
    
    Inputs:
        - pred_probs: predicted probabilities [W, D, L]
        (must sum 1)
        - actual_probs: actual probabilities [W, D, L]
        (binary, must sum 1)
        
    Output:
        - rps_value: RPS for specific game
    '''
    
    cum_probs = np.cumsum(pred_probs)
    cum_actual = np.cumsum(actual_probs)
    
    sum_rps = 0
    for i in range(len(actual_probs)):
        sum_rps += (cum_probs[i] - cum_actual[i])**2
    
    rps_value = sum_rps / (len(actual_probs) - 1)
    
    return rps_value
