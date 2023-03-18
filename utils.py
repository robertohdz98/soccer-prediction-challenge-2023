'''
Custom functions
'''

import pickle
import time
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson


def load_and_process_dataset(dataset_route):
    '''
    Converts date column to pd.datetime format, sorts games from older to more recent,
    adds new column Total Goals and cleans games with Total Goals < 0 (invalid)
    '''
    
    file_format = dataset_route.split(".")[-1]
    
    if file_format == "xlsx":
        start_time = time.time()
        training_set = pd.read_excel(dataset_route)
        print(f"Dataset loaded in {round(time.time() - start_time, 3)} seconds.")
    elif file_format == "parquet":
        start_time = time.time()
        training_set = pd.read_parquet(dataset_route)
        print(f"Dataset loaded in {round(time.time() - start_time, 3)} seconds.")
    else:
        raise Exception("File format is not csv nor parquet.")
    
    # Order dataset by date from older to more recent matches
    training_set["Date"] = pd.to_datetime(training_set["Date"], format="%d/%m/%Y")
    training_set = training_set.sort_values(by=["Date"])
    # Total goals in the match = HS + AS
    training_set["Goals"] = training_set["HS"] + training_set["AS"]
    # Drop invalid games (with negative total goals)
    training_set = training_set.drop(training_set[training_set["Goals"] < 0].index)
    
    return training_set


def seasons_analysis(dataset: pd.DataFrame, plot_goals_per_season=True):
    # Number of games per season
    seasons = dataset["Sea"].value_counts().index.tolist()
    season_games = dataset["Sea"].value_counts().tolist()
    season_games_df = pd.DataFrame({"Season": seasons,
                                    "Number of Games": season_games})
    
    # season_games_df.to_csv("datasets/subsets/season_games.csv", index=None)
    STATS_COLUMNS = ["Sea", "HS", "AS", "GD", "Goals"]
    # Statistics per season
    df_by_season = dataset[STATS_COLUMNS].groupby("Sea").sum()
    
    if plot_goals_per_season is True:
        X_axis = np.arange(len(seasons))

        # Plot Goals per season
        plt.bar(X_axis - 0.1, df_by_season["HS"].to_list(), 0.2, label='HS')
        plt.bar(X_axis + 0.1, df_by_season["AS"].to_list(), 0.2, label='AS')
        
        plt.xticks(X_axis, seasons, rotation=30)
        plt.xlabel("Seasons")
        plt.ylabel("Number of Goals")
        plt.title("Number of Goals per Season")
        plt.legend()
        # plt.savefig("figures/GoalsPerSeason_2.png")
        
        # Plot Avg Goals per season
        plt.bar(X_axis - 0.1, [i / j for i, j in zip(df_by_season["HS"].to_list(), season_games)],
                0.2, label='HS avg')
        plt.bar(X_axis + 0.1, [i / j for i, j in zip(df_by_season["AS"].to_list(), season_games)],
                0.2, label='AS avg')
        
        plt.xticks(X_axis, seasons, rotation=30)
        plt.xlabel("Seasons")
        plt.ylabel("Number of Goals")
        plt.title("Average game goals per season")
        plt.legend()
        # plt.savefig("figures/GoalAvg_Season_2.png")
    
    return season_games_df, df_by_season


def column_stats_by_season(df_by_season: pd.DataFrame, stats_col: str):
    '''Retrieves season with most and least (stat) and the max/min stat.
    
    Args:
        - stat_col: column to extract max/min season
    '''
    
    if stats_col in df_by_season.columns:
        
        max_season = df_by_season[[stats_col]].idxmax()[0]
        max_stat = df_by_season[[stats_col]].max()[0]
        min_season = df_by_season[[stats_col]].idxmin()[0]
        min_stat = df_by_season[[stats_col]].min()[0]
        
        print(f"Season with most {stats_col}: {max_season} "
              f"({max_stat} goals)")
        print(f"Season with least {stats_col}: {min_season} "
              f"({min_stat} goals)")
        
        return max_season, max_stat, min_season, min_stat
    
    else:
        raise ValueError("Column not in df_by_season")
    
   
def search_match(df, team1, team2, order_cares=False):
    
    condition_a = ((df["HT"] == team1) & (df["AT"] == team2))
    condition_b = ((df["HT"] == team2) & (df["AT"] == team1))
    if order_cares is True:
        return df.loc[condition_a]
    else:
        return df.loc[condition_a | condition_b]

 
def get_last_n_matches_result(team_name, n_previous_matches):

    team_df, _, _ = extract_team_df(team_name)
    last_n_games_df = team_df[-n_previous_matches:]

    return last_n_games_df


def extract_team_df(dataset, team_name):
    ''' Extracts games of a specific team from whole training set.
    '''
    
    home_team_df = dataset[dataset["HT"] == team_name]
    away_team_df = dataset[dataset["AT"] == team_name]
        
    team_df = pd.concat([home_team_df, away_team_df]).sort_values(by="Date")
    
    return team_df, home_team_df, away_team_df


def get_stats(df, local=True):
    '''Fills in team statistics dictionary.
    '''
    team_stats = dict()
    if local is True:
        prefix = "home"
        goals_scored_col = "HS"
        goals_against_col = "AS"
        win, loss = "W", "L"
        goals_difference = df["GD"].sum()
        try:
            avg_goals_difference = round(df["GD"].sum() / df.shape[0], 3)
        except ZeroDivisionError:
            avg_goals_difference = np.nan
    else:
        prefix = "away"
        goals_scored_col = "AS"
        goals_against_col = "HS"
        win, loss = "L", "W"
        goals_difference = - df["GD"].sum()
        try:
            avg_goals_difference = - round(df["GD"].sum() / df.shape[0], 3)
        except ZeroDivisionError:
            avg_goals_difference = np.nan

    team_stats[f"{prefix}_games"] = df.shape[0]
    
    team_stats[f"{prefix}_wins"] = df["WDL"].value_counts().get(win, 0)
    team_stats[f"{prefix}_draws"] = df["WDL"].value_counts().get("D", 0)
    team_stats[f"{prefix}_losses"] = df["WDL"].value_counts().get(loss, 0)
    
    try:
        team_stats[f"{prefix}_win_percentage"] = round(df["WDL"].value_counts().get(win, 0) / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_win_percentage"] = np.nan
    try:
        team_stats[f"{prefix}_draw_percentage"] = round(df["WDL"].value_counts().get("D", 0) / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_draw_percentage"] = np.nan
    try:
        team_stats[f"{prefix}_loss_percentage"] = round(df["WDL"].value_counts().get(loss, 0) / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_loss_percentage"] = np.nan
    team_stats[f"{prefix}_goals_scored"] = df[goals_scored_col].sum()
    team_stats[f"{prefix}_goals_against"] = df[goals_against_col].sum()
    team_stats[f"{prefix}_goals_difference"] = goals_difference
    try:
        team_stats[f"{prefix}_avg_goals_scored"] = round(df[goals_scored_col].sum() / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_avg_goals_scored"] = np.nan
    try:
        team_stats[f"{prefix}_avg_goals_against"] = round(df[goals_against_col].sum() / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_avg_goals_against"] = np.nan
    team_stats[f"{prefix}_avg_goals_difference"] = avg_goals_difference

    return team_stats


def complete_team_info(team_name):

    # Filter games only of specified team
    team_df, home_team_df, away_team_df = extract_team_df(team_name)
        
    # Create team dict
    team_stats = dict()

    # Recover home and away teams separately
    home_stats = get_stats(home_team_df, local=True)
    away_stats = get_stats(away_team_df, local=False)

    # Add home and away stats
    team_stats_temp = home_stats | away_stats

    # Add general stats
    team_stats["name"] = team_name
    team_stats["total_games"] = team_df.shape[0]
    team_stats["wins"] = team_stats_temp["home_wins"] + team_stats_temp["away_wins"]
    team_stats["draws"] = team_stats_temp["home_draws"] + team_stats_temp["away_draws"]
    team_stats["losses"] = team_stats_temp["home_losses"] + team_stats_temp["away_losses"]
    team_stats["win_percentage"] = round(team_stats["wins"] / team_df.shape[0], 3)
    team_stats["draw_percentage"] = round(team_stats["draws"] / team_df.shape[0], 3)
    team_stats["loss_percentage"] = round(team_stats["losses"] / team_df.shape[0], 3)
    team_stats["goals_scored"] = team_stats_temp["home_goals_scored"] + team_stats_temp["away_goals_scored"]
    team_stats["goals_against"] = team_stats_temp["home_goals_against"] + team_stats_temp["away_goals_against"]
    team_stats["goals_difference"] = team_stats["goals_scored"] - team_stats["goals_against"]
    team_stats["avg_goals_scored"] = round(team_stats["goals_scored"] / team_df.shape[0], 3)
    team_stats["avg_goals_against"] = round(team_stats["goals_against"] / team_df.shape[0], 3)
    team_stats["avg_goals_difference"] = round(team_stats["goals_difference"] / team_df.shape[0], 3)
    
    team_stats = team_stats | home_stats | away_stats
       
    return team_stats


def search_H2H(dataset, team1, team2):
    
    condition_a = ((dataset["HT"] == team1) & (dataset["AT"] == team2))
    condition_b = ((dataset["HT"] == team2) & (dataset["AT"] == team1))
    
    h2h = dataset.loc[condition_a | condition_b]
    
    return h2h


def get_last_matches_info(team_name, n_previous_matches=5):
    ''' Retrieves info of last N games of a team: last games,
    WDL streak of the team in that games and points (5 last games by default)
    '''

    team_df, _, _ = extract_team_df(team_name)
    last_n_games_df = team_df[-n_previous_matches:]
    
    # Print team streak in last n games
    streak = list()
    for i, game in last_n_games_df.iterrows():
        if (game["WDL"] == "W" and game["HT"] == team_name) or (game["WDL"] == "L" and game["AT"] == team_name):
            streak.append("W")
        elif game["WDL"] == "D":
            streak.append("D")
        else:
            streak.append("L")
    
    # Get points in that streak
    points = streak.count("W") * 3 + streak.count("D")

    return last_n_games_df, streak, points


def get_season_points(dataset, team_name, season="22-23"):
    ''' Computes points of the team in a specific season. If season
    is not provided, computes points from current season (22-23).
    '''
    
    seasons = dataset["Sea"].value_counts().index.tolist()
    
    if season in seasons:
        team_df, _, _ = extract_team_df(team_name)
        season_team_df = team_df[team_df["Sea"] == season]
        season_games = season_team_df.shape[0]
        
        wins = len(season_team_df[(season_team_df['HT'] == team_name) & (season_team_df['WDL'] == "W")]) + \
            len(season_team_df[(season_team_df['AT'] == team_name) & (season_team_df['WDL'] == "L")])
        draws = len(season_team_df[season_team_df['WDL'] == "D"])

        season_points = 3 * wins + draws
        avg_points_season = round(season_points / season_games, 3)
        
        return season_points, avg_points_season
    
    else:
        raise ValueError("Invalid season.")
    
    
def get_leagues(dataset: pd.DataFrame) -> list:
    
    leagues = dataset["Lge"].unique().tolist()
    print(f"Distinct leagues in set: {len(leagues)}")
    
    return leagues


def get_teams_in_league(dataset: pd.DataFrame, league: str) -> list:
    
    leagues = get_leagues(dataset)
    if league not in leagues:
        raise ValueError(f"{league} not in leagues.")
    else:
        h_teams = dataset[dataset["Lge"] == league]["HT"].unique()
        a_teams = dataset[dataset["Lge"] == league]["AT"].unique()
        
        return list(set(h_teams.tolist() + a_teams.tolist()))


with open("teams.pickle", "rb") as f:
    teams_stats_dict = pickle.load(f)


def predict_game(teams: dict, home_team: str, away_team: str):
    ''' Predicts game outcome in terms of exact scores (HS/AS) and
    also WDL probabilities, given two teams in training set (team stats
    should be available to compute game prediction).
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
    
    pr_home, pr_away, pr_draw = 0, 0, 0
    result_probs = list()
    
    for x in range(0, 14):  # number of goals home team (max14)
        for y in range(0, 14):  # number of goals away team (max14)
            p = poisson.pmf(x, lambda_home_team) * poisson.pmf(y, lambda_away_team)
            result_probs.append((x, y, p))
            # if p > 0.01: print(f"Result: {x}-{y}, Prob: {p}")
            if x == y:
                pr_draw += p
            elif x > y:
                pr_home += p
            else:
                pr_away += p
    
    HS, AS, P = max(result_probs, key=itemgetter(2))
      
    # print(f"{home_team}: {pr_home}")
    # print(f"Draw: {pr_draw}")
    # print(f"{away_team}: {pr_away}")
    # print(f"Most probable result: {HS}-{AS} ({P})")
                 
    # print(lambda_home_team, lambda_away_team)
    
    return HS, AS, pr_home, pr_draw, pr_away


def fill_predictions(test_set: pd.DataFrame):
    ''' Fills pd.DF with games as rows in terms of predicted exact
    scores (pr_HS/prd_AS) and prd_WDL probabilities.
    '''
    for i, game in test_set.iterrows():
        
        home_team = game.HT
        away_team = game.AT
        try:
            HS, AS, W, D, L = predict_game(home_team, away_team)
        except:
            continue  # ignore games with teams not present in training set
        
        test_set.at[i, "pr_HS"] = HS
        test_set.at[i, "prd_AS"] = AS
        
        test_set.at[i, "prd_W"] = W
        test_set.at[i, "prd_D"] = D
        test_set.at[i, "prd_L"] = L
        # print(HS, AS, W, D, L)
        
    return test_set
