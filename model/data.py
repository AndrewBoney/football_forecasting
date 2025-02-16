import polars as pl
import math
import sqlite3
from datasets import Dataset

# Define Hugging Face repository configurations
HF_REPO_ID = "YOUR_HF_REPO_ID"  # e.g., "username/football-dataset"
HF_TOKEN = "YOUR_HF_TOKEN"      # Use your Hugging Face access token (or set up environment variable)

def load_fixture_data(db_path: str) -> pl.DataFrame:
    """Load fixture data from SQLite database and sort chronologically."""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT 
                fixture_id, league_id, season, date, round, 
                home_team_id, away_team_id, home_goals, away_goals
            FROM fixtures
        """
        df = pl.read_database(query, conn)
    return df.sort("date")


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
    
    for row in df.iter_rows():
        (fixture_id, league_id, season, date, round_, 
         home_team_id, away_team_id, home_goals, away_goals) = row

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
            "away_defense_rating": away_defense
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


def push_dataset_to_huggingface(dataset_df: pl.DataFrame, repo_id: str, hf_token: str = None) -> None:
    """
    Convert a Polars DataFrame to a Hugging Face Dataset and push it to the Hub.
    
    Args:
        dataset_df (pl.DataFrame): The dataset to be uploaded.
        repo_id (str): The repository identifier (e.g., "username/dataset-name").
        hf_token (str): Your Hugging Face access token.
    """
    # Convert Polars DataFrame to pandas
    df_pandas = dataset_df.to_pandas()
    
    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df_pandas)
    
    # Push dataset to Hugging Face Hub
    hf_dataset.push_to_hub(repo_id, token=hf_token)
    print(f"Dataset successfully pushed to Hugging Face repository: {repo_id}")


def main():
    """Main workflow executing the rating calculation process."""
    # Configuration
    DB_PATH = "football.db"
    K = 0.05
    HOME_ADVANTAGE = 0.1

    # Load and prepare data
    fixtures_df = load_fixture_data(DB_PATH)
    
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
    history_df = pl.DataFrame(ratings_history)
    
    # Write dataset to Hugging Face Hub
    push_dataset_to_huggingface(history_df, HF_REPO_ID, HF_TOKEN)
    
    return history_df


if __name__ == "__main__":
    final_history = main()