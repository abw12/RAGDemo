from pathlib import Path
from typing import Dict, List
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

def load_dataframes() -> Dict[str,pd.DataFrame]:
    """
    Load all CSVs in data/ as pandas DataFrames.
    Key = file stem, e.g. 'batter_player_stats'.
    """
    dfs:Dict[str,pd.DataFrame] = {}
    for csv_path in DATA_DIR.glob("*.csv"):
        name = csv_path.stem
        print(f"[TABULAR] Loading DataFrame: {name}")
        df = pd.read_csv(csv_path)
        dfs[name] = df
    return dfs

def find_player_names_in_question(
        question: str,
        dfs: Dict[str,pd.DataFrame],
        player_column: str = "player_name"
) -> List[str]:
    
    """
    Very simple heuristic:
    - Look for exact player_name substrings from any DF's player_name column
      inside the question (case-insensitive).
    - Works generically as long as the column exists.
    """

    q = question.lower()
    found:List[str] = []

    for _, df in dfs.items():
        if player_column not in df.columns:
            continue
        
        for raw_name in df[player_column].dropna().unique():
            name = str(raw_name)
            if name and name.lower() in q:
                if name not in found:
                    found.append(name)
    return found

def get_player_rows(
        dfs:Dict[str,pd.DataFrame],
        player_name: str,
        player_column: str = "player_name",
) -> Dict[str,pd.DataFrame]:
    """
    Return all rows for a given player_name from all DataFrames that have that column.
    Result: {table_name: filtered_df}
    """ 

    result: Dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        if player_column in df.columns:
            subset = df[df[player_column] == player_name]
            if not subset.empty:
                result[name] = subset
    return result
