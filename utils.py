'''
Custom functions
'''


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_process_dataset(dataset_route: str) -> pd.DataFrame:
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


TRAINING_SET = pd.read_parquet("datasets/processed/training_set_processed.parquet")


def seasons_analysis(dataset: pd.DataFrame = TRAINING_SET,
                     plot_goals_per_season: bool = True):
    ''' Computes some stats by each season present in dataset.
    '''
    
    STATS_COLUMNS = ["Sea", "HS", "AS", "GD", "Goals"]
    
    # Statistics per season
    df_by_season = dataset[STATS_COLUMNS].groupby("Sea").sum()
    df_by_season["Games"] = dataset.groupby("Sea").size()
    season_games = df_by_season["Games"].tolist()
    
    if plot_goals_per_season is True:
        plt.figure()
        plt.subplot(211, title="Number of Goals per Season")
        seasons = df_by_season.index.tolist()
        X_axis = np.arange(len(seasons))

        # Plot Goals per season
        plt.bar(X_axis - 0.1, df_by_season["HS"].to_list(), 0.2, label='HS')
        plt.bar(X_axis + 0.1, df_by_season["AS"].to_list(), 0.2, label='AS')
        plt.xticks(X_axis, seasons, rotation=30)
        plt.ylabel("Number of Goals")
        plt.legend()
        plt.tight_layout(pad=3.0)
        
        # Plot Avg Goals per season
        plt.subplot(212, title="Average game goals per season")
        plt.bar(X_axis - 0.1,
                [i / j for i, j in zip(df_by_season["HS"].to_list(), season_games)],
                0.2, label='HS avg')
        plt.bar(X_axis + 0.1,
                [i / j for i, j in zip(df_by_season["AS"].to_list(), season_games)],
                0.2, label='AS avg')
        plt.xticks(X_axis, seasons, rotation=30)
        plt.xlabel("Seasons")
        plt.ylabel("Number of Goals")
        plt.legend()
        
        plt.savefig("figures/Seasons_Goals.png")
    
    return df_by_season


def column_stats_by_season(dataset: pd.DataFrame, stats_col: str):
    '''Retrieves season with most and least (stat) and the max/min stat.
    
    Args:
        - stat_col: column to extract max/min season
    '''
    
    df_by_season = seasons_analysis(dataset, plot_goals_per_season=False)
    
    if stats_col in df_by_season.columns:
        
        max_season = df_by_season[[stats_col]].idxmax()[0]
        max_stat = df_by_season[[stats_col]].max()[0]
        min_season = df_by_season[[stats_col]].idxmin()[0]
        min_stat = df_by_season[[stats_col]].min()[0]
        
        print(f"Season with most {stats_col}: {max_season} "
              f"({max_stat})")
        print(f"Season with least {stats_col}: {min_season} "
              f"({min_stat})")
    
    else:
        raise ValueError("Column not in df_by_season")


# ### GENERAL FUNCTIONS FOR DATASET EXPLORATORY ANALYSIS # ###
def get_all_teams(dataset: pd.DataFrame) -> list:
    ''' Gets all distinct teams present in dataset.
    '''
    h_teams = dataset["HT"].unique()
    a_teams = dataset["AT"].unique()
    
    teams = list(set(h_teams.tolist() + a_teams.tolist()))
    print(f"Distinct teams in dataset: {len(teams)}")
    
    return teams

   
def get_leagues(dataset: pd.DataFrame) -> list:
    ''' Gets distinct leagues in dataset.
    '''
    
    leagues = dataset["Lge"].unique().tolist()
    print(f"Distinct leagues in set: {len(leagues)}")
    
    return leagues


def get_teams_in_league(dataset: pd.DataFrame, league: str) -> list:
    ''' Gets distinct teams in specific league of dataset.
    '''
    
    leagues = get_leagues(dataset)
    if league not in leagues:
        raise ValueError(f"{league} not in leagues.")
    else:
        h_teams = dataset[dataset["Lge"] == league]["HT"].unique()
        a_teams = dataset[dataset["Lge"] == league]["AT"].unique()
        
        league_teams = list(set(h_teams.tolist() + a_teams.tolist()))
        print(f"Distinct teams in league: {len(league_teams)}")
        
        return league_teams
    
    
# ### INDIVIDUAL TEAM FUNCTIONS # ###

def search_H2H(dataset: pd.DataFrame,
               team1: str, team2: str,
               order_cares: bool = False):
    ''' Gets H2H games between specified teams in dataset
    (order is not important).
    
    Inputs:
        - dataset: DF with historic of games
        - team1, team2: names of teams to look for H2H
        - order_cares: if True, searchs only for H2H where
        team1=HT and team2=AT (default: False)
    '''
    
    condition_a = ((dataset["HT"] == team1) & (dataset["AT"] == team2))
    condition_b = ((dataset["HT"] == team2) & (dataset["AT"] == team1))
    
    if order_cares is True:
        return dataset.loc[condition_a]
    else:
        h2h = dataset.loc[condition_a | condition_b]
        return h2h


def get_team_last_matches(dataset: pd.DataFrame,
                          team_name: str, n_previous_games: int = 5):
    ''' Retrieves info of last N games of a team: last games,
    WDL streak of the team in that games and points.
    
    Inputs:
        - team_name: name of the team
        - n_previous_games: number of previous games to look
        
    Outputs:
        - last_n_games_df: DF with filtered N last games of team
        - streak: list of ordered last N games of team [W, D, L]
        - points: number of awarded points of team in last N games
    '''

    team_df, _, _ = extract_team_df(team_name, dataset)
    last_n_games_df = team_df[-n_previous_games:]
    
    # Print team streak in last n games
    streak = list()
    for i, game in last_n_games_df.iterrows():
        if (game["WDL"] == "W" and game["HT"] == team_name
                or game["WDL"] == "L" and game["AT"] == team_name):
            streak.append("W")
        elif game["WDL"] == "D":
            streak.append("D")
        else:
            streak.append("L")
    
    # Get points in that streak
    points = streak.count("W") * 3 + streak.count("D")

    return last_n_games_df, streak, points


def get_team_season_points(dataset: pd.DataFrame,
                           team_name: str, season: str = "22-23"):
    ''' Computes points of the team in a specific season. If season
    is not provided, computes points from current season (22-23).
    '''
    
    seasons = dataset["Sea"].value_counts().index.tolist()
    
    if season not in seasons:
        raise ValueError("Invalid season.")
    
    team_df, _, _ = extract_team_df(team_name)
    season_team_df = team_df[team_df["Sea"] == season]
    season_games = season_team_df.shape[0]
    
    # Get number of wins and draws
    wins_h = len(season_team_df[(season_team_df['HT'] == team_name)
                                & (season_team_df['WDL'] == "W")])
    wins_a = len(season_team_df[(season_team_df['AT'] == team_name)
                                & (season_team_df['WDL'] == "L")])
    draws = len(season_team_df[season_team_df['WDL'] == "D"])

    season_points = 3 * (wins_h + wins_a) + draws
    avg_points_season = round(season_points / season_games, 3)
    
    return season_points, avg_points_season


#########################################################################
# ############### FUNCTIONS TO RECORD TEAMS STATISTICS ################ #
#########################################################################


def extract_team_df(team_name: str, dataset: pd.DataFrame):
    ''' Extracts all games of a specific team in specific dataset.
    
    Input:
        - team_name: name of the team in the dataset
        - dataset: pd.DF with historic of games
    Outputs:
        - team_df: pd.DF with all games of that team
        - home_team_df: pd.DF only with games of the team played at home
        - away_team_df: pd.DF only with games of the team played away
    '''
    
    home_team_df = dataset[dataset["HT"] == team_name]
    away_team_df = dataset[dataset["AT"] == team_name]
        
    team_df = pd.concat([home_team_df, away_team_df]).sort_values(by="Date")
    
    return team_df, home_team_df, away_team_df


def get_stats(df: pd.DataFrame, local: bool = True) -> dict:
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
        team_stats[f"{prefix}_win_percentage"] = round(df["WDL"].value_counts().get(win, 0)
                                                       / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_win_percentage"] = np.nan
    try:
        team_stats[f"{prefix}_draw_percentage"] = round(df["WDL"].value_counts().get("D", 0)
                                                        / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_draw_percentage"] = np.nan
    try:
        team_stats[f"{prefix}_loss_percentage"] = round(df["WDL"].value_counts().get(loss, 0)
                                                        / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_loss_percentage"] = np.nan
        
    team_stats[f"{prefix}_goals_scored"] = df[goals_scored_col].sum()
    team_stats[f"{prefix}_goals_against"] = df[goals_against_col].sum()
    team_stats[f"{prefix}_goals_difference"] = goals_difference
    
    try:
        team_stats[f"{prefix}_avg_goals_scored"] = round(df[goals_scored_col].sum()
                                                         / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_avg_goals_scored"] = np.nan
    try:
        team_stats[f"{prefix}_avg_goals_against"] = round(df[goals_against_col].sum()
                                                          / df.shape[0], 3)
    except ZeroDivisionError:
        team_stats[f"{prefix}_avg_goals_against"] = np.nan
    team_stats[f"{prefix}_avg_goals_difference"] = avg_goals_difference

    return team_stats


def complete_team_info(team_name: str) -> dict:
    ''' Completes all defined statistics for a given team.
    '''

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
    team_stats["wins"] = team_stats_temp["home_wins"] + \
        team_stats_temp["away_wins"]
    team_stats["draws"] = team_stats_temp["home_draws"] + \
        team_stats_temp["away_draws"]
    team_stats["losses"] = team_stats_temp["home_losses"] + \
        team_stats_temp["away_losses"]
        
    team_stats["win_percentage"] = round(team_stats["wins"]
                                         / team_df.shape[0], 3)
    team_stats["draw_percentage"] = round(team_stats["draws"]
                                          / team_df.shape[0], 3)
    team_stats["loss_percentage"] = round(team_stats["losses"]
                                          / team_df.shape[0], 3)
    
    team_stats["goals_scored"] = team_stats_temp["home_goals_scored"] + \
        team_stats_temp["away_goals_scored"]
    team_stats["goals_against"] = team_stats_temp["home_goals_against"] + \
        team_stats_temp["away_goals_against"]
    team_stats["goals_difference"] = team_stats["goals_scored"] - \
        team_stats["goals_against"]
        
    team_stats["avg_goals_scored"] = round(team_stats["goals_scored"]
                                           / team_df.shape[0], 3)
    team_stats["avg_goals_against"] = round(team_stats["goals_against"]
                                            / team_df.shape[0], 3)
    team_stats["avg_goals_difference"] = round(team_stats["goals_difference"]
                                               / team_df.shape[0], 3)
    
    team_stats = team_stats | home_stats | away_stats
       
    return team_stats
