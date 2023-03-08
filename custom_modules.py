import numpy as np
import pandas as pd

# TRAINING SET CURATED
TRAINING_SET = pd.read_parquet("datasets/processed/training_set_processed.parquet")


def load_and_process_dataset(training_set_route):
    
    file_format = training_set_route.split(".")[-1]
    
    if file_format == "csv":
        training_set = pd.read_excel(training_set_route)
    elif file_format == "parquet":
        training_set = pd.read_parquet(training_set_route)
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


def extract_team_df(team_name):
    ''' Extracts games of a specific team.
    '''
    
    home_team_df = TRAINING_SET[TRAINING_SET["HT"] == team_name]
    away_team_df = TRAINING_SET[TRAINING_SET["AT"] == team_name]
        
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
