import polars as pl

import math
import sqlite3
import subprocess
import tempfile
import os

from typing import Optional, List
from datasets import Dataset
from huggingface_hub import HfApi


def load_sqlite_data(query: str, db_path: str = 'football.db') -> pl.DataFrame:
    """Load data from SQLite database using a SQL query."""
    with sqlite3.connect(db_path) as conn:
        df = pl.read_database(query, conn)
    return df


def load_fixture_data(cols: Optional[List[str]] = None, db_path: str = "football.db") -> pl.DataFrame:
    """Load fixture data from SQLite database and sort chronologically."""
    if cols:
        cols_str = ", ".join(cols)
        query = f"SELECT {cols_str} FROM fixtures"
    else:
        query = "SELECT * FROM fixtures"

    df = load_sqlite_data(query, db_path)
    return df.sort("date")


def load_team_data(db_path: str = "football.db", cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Load team data from SQLite database."""
    if cols:
        cols_str = ", ".join(cols)
        query = f"SELECT {cols_str} FROM teams"
    else:
        query = "SELECT * FROM teams"
    
    return load_sqlite_data(query, db_path)

def initialize_ratings(df: pl.DataFrame) -> tuple[dict, dict]:
    """Initialize attack and defense ratings for all teams."""
    home_teams = df["home_team_id"].unique().to_list()
    away_teams = df["away_team_id"].unique().to_list()
    teams = set(home_teams + away_teams)
    return (
        {team: 0.0 for team in teams},  # attack_ratings
        {team: 0.0 for team in teams}   # defense_ratings
    )


def calculate_expected_goals(home_attack: float, away_defense: float, 
                             away_attack: float, home_defense: float,
                             home_advantage: float) -> tuple[float, float]:
    """Calculate expected goals using exponential model."""
    expected_home = math.exp(home_attack - away_defense + home_advantage)
    expected_away = math.exp(away_attack - home_defense)
    return expected_home, expected_away


def update_ratings(
    attack_ratings: dict,
    defense_ratings: dict,
    home_team: int,
    away_team: int,
    home_goals: int,
    away_goals: int,
    expected_home: float,
    expected_away: float,
    K: float
) -> None:
    """Update ratings in-place based on match results."""
    # Update home team ratings
    attack_ratings[home_team] += K * (home_goals - expected_home)
    defense_ratings[home_team] += K * (expected_away - away_goals)
    
    # Update away team ratings
    attack_ratings[away_team] += K * (away_goals - expected_away)
    defense_ratings[away_team] += K * (expected_home - home_goals)


def process_matches(
    df: pl.DataFrame,
    attack_ratings: dict,
    defense_ratings: dict,
    K: float,
    home_advantage: float
) -> list[dict]:
    """Process all matches to update ratings and return history."""
    history = []

    nums = {id_ : 0 for id_ in attack_ratings.keys()}
    for row in df.iter_rows():
        (fixture_id, league_id, season, date, round_, 
         home_team_id, away_team_id, home_goals, away_goals) = row

        # iterate nums
        nums[home_team_id] += 1
        nums[away_team_id] += 1

        # Get current ratings
        home_attack = attack_ratings[home_team_id]
        home_defense = defense_ratings[home_team_id]
        away_attack = attack_ratings[away_team_id]
        away_defense = defense_ratings[away_team_id]

        # Calculate expected goals
        expected_home, expected_away = calculate_expected_goals(
            home_attack, away_defense, away_attack, home_defense, home_advantage
        )

        # Record history before updates
        history.append({
            "fixture_id": fixture_id,
            "season": season,
            "league_id": league_id,
            "round": round_,
            "date": date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "expected_home": expected_home,
            "expected_away": expected_away,
            "home_attack_rating": home_attack,
            "home_defense_rating": home_defense,
            "away_attack_rating": away_attack,
            "away_defense_rating": away_defense,
            "home_team_fixture_num" : nums[home_team_id],
            "away_team_fixture_num" : nums[away_team_id]
        })

        # Update ratings based on match outcome
        update_ratings(
            attack_ratings,
            defense_ratings,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals,
            expected_home,
            expected_away,
            K
        )

    return history

def add_date_columns(df: pl.DataFrame, date_col : str) -> pl.DataFrame:
    return (df
        .with_columns(
            pl.col(date_col).str.strptime(pl.Date, format = "%Y-%m-%dT%H:%M:%S%z")
        )
        .with_columns(
            pl.col(date_col).dt.weekday().alias("day_of_week"),
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.year().alias("year")
        )            
    )

def add_historical_columns(df: pl.DataFrame) -> pl.DataFrame:
    long_data = (df
        .unpivot(
            index = ["fixture_id", "season", "year", "round_int", "expected_home", "expected_away", "home_goals", "away_goals"], 
            on = ["home_team_id", "away_team_id"]
        )
        .with_columns(
            pl.when(pl.col("variable") == pl.lit("home_team_id")).then(pl.col("expected_home")).otherwise(pl.col("expected_away")).alias("expected_goals"),
            pl.when(pl.col("variable") == pl.lit("home_team_id")).then(pl.col("expected_away")).otherwise(pl.col("expected_home")).alias("expected_conceded"),
            pl.when(pl.col("variable") == pl.lit("home_team_id")).then(pl.col("home_goals")).otherwise(pl.col("away_goals")).alias("actual_goals"),
            pl.when(pl.col("variable") == pl.lit("home_team_id")).then(pl.col("away_goals")).otherwise(pl.col("home_goals")).alias("actual_conceded"),
            pl.when(pl.col("variable") == pl.lit("home_team_id")).then(pl.lit(True)).otherwise(pl.lit(False)).alias("is_home")
        )
        .rename({"value" : "team_id"})
        .select(
            "fixture_id", "season", "year", "round_int", "team_id", "is_home", "expected_goals", "expected_conceded", "actual_goals", "actual_conceded"
        )
        .sort(["season", "team_id", "round_int"])
    )

    cols = ["expected_goals", "expected_conceded", "actual_goals", "actual_conceded"]
    long_data = (long_data
        .with_columns(*[
            [
                pl.col(c).shift(i).over("season", "team_id").alias(f"{c}_prev{i}")
                for i in range(1, 5+1)
            ]
            for c in cols
        ])
        .with_columns([
            pl.concat_list([pl.col(f"{c}_prev{i}") for i in range(1, 5+1)]).alias(f"{c}_prev_list")
            for c in cols
        ])
        .select("fixture_id", "team_id", *[f"{c}_prev_list" for c in cols])
    )

    df = (df
        .join(
            long_data.rename({"team_id": "home_team_id"}), 
            how = "left", 
            on=["fixture_id", "home_team_id"]
        )
        .join(
            long_data.rename({"team_id": "away_team_id"}), 
            how = "left", 
            on=["fixture_id", "away_team_id"],
            suffix="_away"
        )
        .rename({f"{col}_prev_list" : f"{col}_prev_list_home" for col in cols})
    )

    return df

def prepare(
    DB_PATH: str = "football.db",
    K: float = 0.05,
    HOME_ADVANTAGE: float = 0.1
) -> pl.DataFrame:
    """Prepare the dataset by calculating ratings and history."""
    # Load and prepare data
    fixtures_df = load_fixture_data(
        cols = ['fixture_id', 'league_id', 'season', 'date', 'round', 'home_team_id', 'away_team_id', 'home_goals', 'away_goals'],
        db_path=DB_PATH
    )
    
    # Initialize ratings
    attack_ratings, defense_ratings = initialize_ratings(fixtures_df)
    
    # Process matches and track history
    ratings_history = process_matches(
        fixtures_df,
        attack_ratings,
        defense_ratings,
        K,
        HOME_ADVANTAGE
    )
    
    # Convert history to DataFrame for analysis
    df = pl.DataFrame(ratings_history)

    # Add any additional processing
    ## Date Columns
    df = add_date_columns(df, "date")
    
    ## Create integer round
    df = df.with_columns(
        pl.col("round").str.split(" - ").list.get(1).cast(pl.Int64).alias("round_int")
    ) 

    ## Make historical
    df = add_historical_columns(df)

    return df


def push_dataset_to_huggingface(
    dataset_df: pl.DataFrame, 
    repo_id: str,
    filter_n_games: int = 20,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> None:
    """
    Convert a Polars DataFrame to a Hugging Face Dataset and push it to the Hub.
    
    Args:
        dataset_df (pl.DataFrame): The dataset to be uploaded.
        repo_id (str): The repository identifier (e.g., "username/dataset-name").
        filter_n_games (int): Remove games without n_games warmup. 
    """
    # Filter dataframe
    dataset_df = dataset_df.filter(
        pl.col("home_team_fixture_num") >= filter_n_games, 
        pl.col("away_team_fixture_num") >= filter_n_games
    )

    # Convert Polars DataFrame to pandas DataFrame
    df_pandas = dataset_df.to_pandas()
    
    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df_pandas)
    
    train = hf_dataset.train_test_split(test_size = test_size + val_size, seed = 42)
    test = train["test"].train_test_split(test_size = val_size / (val_size + test_size), seed = 42)

    datasets = {
        "train" : train["train"],
        "test" : test["train"],
        "val" : test["test"]
    }

    for split, data in datasets.items():
        # Push dataset to Hugging Face Hub
        data.push_to_hub(repo_id, split = split)

def get_git_info() -> str:
    """
    Get git commit info of the current working repository.
    Returns a string containing the commit hash, branch name, and remote URL.
    """
    try:
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        # Get current branch name
        branch_name = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        # Get remote URL
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        return f"Git commit: `{commit_hash}` (branch: `{branch_name}`) (remote: `{remote_url}`)"
    except Exception:
        return "Git info unavailable."

def push_model_card(dataset_df: pl.DataFrame, repo_id: str, hf_token: str = None) -> None:
    """
    Create and push a model card (README.md) to the Hugging Face Hub repository.
    
    The model card will include:
      - A description of the dataset.
      - The leagues and time periods included in the dataset.
      - The git info of the current working repository.
    """
    # Description of the dataset
    description = (
        "This dataset contains football fixtures with match details and the evolving ratings "
        "of teams based on fixture outcomes. The ratings are updated using an exponential model."
    )
    
    # Extract leagues and time periods from the dataset
    leagues = sorted(set(dataset_df["league_id"].unique().to_list()))
    try:
        # Assuming date is stored in a sortable format (e.g., YYYY-MM-DD)
        min_date = dataset_df["date"].min()
        max_date = dataset_df["date"].max()
    except Exception:
        min_date, max_date = "N/A", "N/A"

    # Git info from current repository
    git_info = get_git_info()
    
    model_card = f"""# Football Ratings Dataset

{description}

## Leagues Included
The dataset includes the following leagues:  
{', '.join(str(l) for l in leagues)}

## Time Period
Data covers from **{min_date}** to **{max_date}**.

## Git Repository Information
{git_info}

---

This dataset is automatically generated using football match fixtures and updated ratings calculated using a simple exponential model.
    """
    
    # Write the model card content to a temporary file and push it as README.md to the HF Hub.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as tmp:
        tmp.write(model_card)
        tmp_path = tmp.name

    # Use the HF API to upload the file as README.md to the dataset repository
    api = HfApi()
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=hf_token,
        repo_type="dataset"
    )
    print("Model card (README.md) successfully pushed to the Hugging Face repository.")


def push(
    df: pl.DataFrame,
    HF_REPO_ID: str = "AndyB/football_fixtures",
    HF_TOKEN: str = os.getenv("HF_TOKEN")
) -> None:
    """Push the dataset and model card to the Hugging Face Hub."""
    # Write dataset to Hugging Face Hub
    push_dataset_to_huggingface(df, HF_REPO_ID)
    
    # Push model card (README.md) to Hugging Face Hub
    push_model_card(df, HF_REPO_ID, HF_TOKEN)


def prepare_and_push(
    DB_PATH: str = "football.db",
    K: float = 0.05,
    HOME_ADVANTAGE: float = 0.1,
    HF_REPO_ID: str = "AndyB/football_fixtures",
    HF_TOKEN: str = os.getenv("HF_TOKEN")
):
    """Main workflow executing the rating calculation process and pushing to HF Hub."""
    # Prepare the dataset
    df = prepare(DB_PATH, K, HOME_ADVANTAGE)
    
    # Push the dataset and model card to Hugging Face Hub
    push(df, HF_REPO_ID, HF_TOKEN)
