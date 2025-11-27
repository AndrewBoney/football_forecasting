import json
import sqlite3
import os
import requests

from datetime import datetime
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class RawDataStore:
    """Handles raw API response storage"""
    
    def __init__(self, db_path: str = 'football.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Raw API response storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_api_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                parameters TEXT NOT NULL,
                response_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed_flag BOOLEAN DEFAULT 0,
                error_info TEXT
            )
        """)
        
        # Pipeline state tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_state (
                league_id INTEGER,
                season INTEGER,
                endpoint TEXT,
                last_updated DATETIME,
                PRIMARY KEY (league_id, season, endpoint)
            )
        """)
        
        self.conn.commit()

    def save_raw_response(self, endpoint: str, params: dict, response_json: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO raw_api_responses (endpoint, parameters, response_json)
            VALUES (?, ?, ?)
        """, (endpoint, json.dumps(params), response_json))
        self.conn.commit()

    def get_last_update(self, league_id: int, season: int, endpoint: str) -> Optional[datetime]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT last_updated FROM pipeline_state
            WHERE league_id = ? AND season = ? AND endpoint = ?
        """, (league_id, season, endpoint))
        result = cursor.fetchone()
        return datetime.fromisoformat(result[0]) if result else None

    def update_state(self, league_id: int, season: int, endpoint: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pipeline_state 
            (league_id, season, endpoint, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (league_id, season, endpoint))
        self.conn.commit()

class APIClient:
    """Handles API communication with retry logic"""
    
    def __init__(self, api_key: str):
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
            'x-rapidapi-key': api_key
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_data(self, endpoint: str, params: dict) -> dict:
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

class FootballPipeline:
    """Orchestrates data collection and processing"""
    
    def __init__(self, api_key: str, db_path: str = 'football.db'):
        self.api = APIClient(api_key)
        self.raw_store = RawDataStore(db_path)
        self.processors = {
            'leagues': self._process_leagues,
            'fixtures': self._process_fixtures,
            'teams': self._process_teams,
            'players': self._process_players
        }

    def run_collection(self, leagues: List[Dict]):
        """Main entry point for data collection"""
        for league_config in leagues:
            league_id = league_config['id']
            for season in league_config['seasons']:
                self._process_league_season(league_id, season)

    def _process_league_season(self, league_id: int, season: int):
        """Process all endpoints for a league-season combination"""
        endpoints = [
            ('leagues', {'id': league_id}),
            ('fixtures', {'league': league_id, 'season': season}),
            ('teams', {'league': league_id, 'season': season}),
            ('players', {'league': league_id, 'season': season}),
        ]

        for endpoint, params in endpoints:
            last_update = self.raw_store.get_last_update(league_id, season, endpoint)
            if self._needs_refresh(endpoint, last_update):
                self._fetch_and_store(endpoint, params, league_id, season)

    def _needs_refresh(self, endpoint: str, last_update: Optional[datetime]) -> bool:
        """Determine if data needs refreshing based on update frequency"""
        if not last_update:
            return True
        refresh_intervals = {
            'leagues': 7,    # Days
            'fixtures': 1,
            'teams': 7,
            'players': 30
        }
        return (datetime.now() - last_update).days > refresh_intervals.get(endpoint, 7)

    def _fetch_and_store(self, endpoint: str, params: dict, league_id: int, season: int):
        """Fetch data from API and store raw response"""
        try:
            data = self.api.fetch_data(endpoint, params)
            self.raw_store.save_raw_response(
                endpoint=endpoint,
                params=params,
                response_json=json.dumps(data)
            )
            self.raw_store.update_state(league_id, season, endpoint)
        except Exception as e:
            print(f"Error processing {endpoint} for league {league_id}: {str(e)}")
            raise

    def run_processing(self):
        """Process all unprocessed raw responses"""
        cursor = self.raw_store.conn.cursor()
        cursor.execute("""
            SELECT id, endpoint, response_json 
            FROM raw_api_responses 
            WHERE processed_flag = 0
        """)
        
        for row in cursor.fetchall():
            response_id, endpoint, response_json = row
            try:
                data = json.loads(response_json)
                self.processors[endpoint](data['response'])
                self._mark_processed(response_id)
            except Exception as e:
                self._mark_error(response_id, str(e))

    # Processing methods for each endpoint
    def _process_leagues(self, data: List[Dict]):
        cursor = self.raw_store.conn.cursor()
        
        # Create normalized tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leagues (
                league_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                logo_url TEXT,
                country_name TEXT,
                country_code TEXT,
                country_flag TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seasons (
                league_id INTEGER,
                year INTEGER NOT NULL,
                start_date DATE,
                end_date DATE,
                is_current BOOLEAN,
                FOREIGN KEY(league_id) REFERENCES leagues(league_id)
            )
        """)

        for league_data in data:
            try:
                # Extract league information
                league = league_data['league']
                country = league_data['country']
                seasons = league_data['seasons']

                # Insert league (or ignore if already exists)
                cursor.execute("""
                    INSERT OR IGNORE INTO leagues 
                    VALUES (:id, :name, :type, :logo, :country_name, :country_code, :country_flag)
                """, {
                    'id': league['id'],
                    'name': league['name'],
                    'type': league['type'],
                    'logo': league.get('logo'),
                    'country_name': country.get('name'),
                    'country_code': country.get('code'),
                    'country_flag': country.get('flag')
                })

                # Insert seasons
                for season in seasons:
                    cursor.execute("""
                        INSERT OR IGNORE INTO seasons 
                        (league_id, year, start_date, end_date, is_current)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        league['id'],
                        season['year'],
                        season['start'],
                        season['end'],
                        season['current']
                    ))

            except KeyError as e:
                print(f"Missing key in league data: {e}")
                continue
                
        self.raw_store.conn.commit()

    def _process_fixtures(self, data: List[Dict]):
        cursor = self.raw_store.conn.cursor()
        
        # Create fixtures table without venue reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id INTEGER PRIMARY KEY,
                league_id INTEGER,
                season INTEGER,
                date TIMESTAMP,
                referee TEXT,
                status TEXT,
                round TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_goals INTEGER,
                away_goals INTEGER,
                home_winner BOOLEAN,
                away_winner BOOLEAN,
                FOREIGN KEY(league_id) REFERENCES leagues(league_id),
                FOREIGN KEY(home_team_id) REFERENCES teams(team_id),
                FOREIGN KEY(away_team_id) REFERENCES teams(team_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_scores (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER,
                period TEXT CHECK(period IN ('halftime', 'fulltime', 'extratime', 'penalty')),
                home_goals INTEGER,
                away_goals INTEGER,
                FOREIGN KEY(fixture_id) REFERENCES fixtures(fixture_id)
            )
        """)

        for fixture_data in data:
            try:
                # Extract nested data
                fixture = fixture_data['fixture']
                league = fixture_data['league']
                teams = fixture_data['teams']
                goals = fixture_data['goals']
                score = fixture_data['score']

                # Insert main fixture (without venue)
                cursor.execute("""
                    INSERT OR REPLACE INTO fixtures 
                    VALUES (
                        :fixture_id,
                        :league_id,
                        :season,
                        :date,
                        :referee,
                        :status,
                        :round,
                        :home_team_id,
                        :away_team_id,
                        :home_goals,
                        :away_goals,
                        :home_winner,
                        :away_winner
                    )
                """, {
                    'fixture_id': fixture['id'],
                    'league_id': league['id'],
                    'season': league['season'],
                    'date': fixture['date'],
                    'referee': fixture.get('referee'),
                    'status': fixture['status']['long'],
                    'round': league.get('round'),
                    'home_team_id': teams['home']['id'],
                    'away_team_id': teams['away']['id'],
                    'home_goals': goals['home'],
                    'away_goals': goals['away'],
                    'home_winner': teams['home']['winner'],
                    'away_winner': teams['away']['winner']
                })

                # Insert score periods
                for period in ['halftime', 'fulltime', 'extratime', 'penalty']:
                    period_score = score.get(period)
                    if period_score and period_score['home'] is not None:
                        cursor.execute("""
                            INSERT INTO match_scores 
                            (fixture_id, period, home_goals, away_goals)
                            VALUES (?, ?, ?, ?)
                        """, (
                            fixture['id'],
                            period,
                            period_score['home'],
                            period_score['away']
                        ))

            except KeyError as e:
                print(f"Missing key in fixture data: {e}")
                continue
                
        self.raw_store.conn.commit()

    def _process_teams(self, data: List[Dict]):
        cursor = self.raw_store.conn.cursor()
        
        # Create teams table with minimal venue reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                code TEXT,
                country TEXT,
                founded INTEGER,
                national BOOLEAN,
                logo_url TEXT,
                venue_id INTEGER
            )
        """)

        for team_data in data:
            try:
                team = team_data['team']
                venue = team_data.get('venue', {})
                
                cursor.execute("""
                    INSERT OR REPLACE INTO teams 
                    VALUES (
                        :team_id,
                        :name,
                        :code,
                        :country,
                        :founded,
                        :national,
                        :logo,
                        :venue_id
                    )
                """, {
                    'team_id': team['id'],
                    'name': team['name'],
                    'code': team.get('code'),
                    'country': team.get('country'),
                    'founded': team.get('founded'),
                    'national': team.get('national', False),
                    'logo': team.get('logo'),
                    'venue_id': venue.get('id')
                })

            except KeyError as e:
                print(f"Missing key in team data: {e}")
                continue
                
        self.raw_store.conn.commit()
        
    def _process_players(self, data: List[Dict]):
        # Similar implementation for players
        pass

    def _mark_processed(self, response_id: int):
        self.raw_store.conn.execute(
            "UPDATE raw_api_responses SET processed_flag = 1 WHERE id = ?",
            (response_id,)
        )
        self.raw_store.conn.commit()

    def _mark_error(self, response_id: int, error_message: str):
        self.raw_store.conn.execute(
            "UPDATE raw_api_responses SET error_info = ? WHERE id = ?",
            (error_message, response_id)
        )
        self.raw_store.conn.commit()
