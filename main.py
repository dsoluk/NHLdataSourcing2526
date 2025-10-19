import os
import sys
from datetime import datetime
from typing import Optional
import re
import json

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


def get_downloads_dir() -> str:
    """Return the user's Downloads folder path in a cross-user, Windows-friendly way."""
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    # Fallback if Downloads doesn't exist
    if not os.path.isdir(downloads):
        downloads = home
    return downloads


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns and normalize names."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(p).strip() for p in tup if p is not None and str(p).strip() != ""]
            col = "_".join(parts) if parts else ""
            new_cols.append(col)
        df.columns = new_cols
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Remove common Hockey-Reference placeholder names
    df.columns = [c.replace("Unnamed: ", "").replace("level_0", "").strip(" _") for c in df.columns]
    # De-duplicate columns while preserving the first occurrence
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def _clean_string_series(s: pd.Series) -> pd.Series:
    """Clean a string/object series: trim, remove footnote markers, commas, %.
    This prepares for potential numeric conversion.
    """
    s = s.astype(str)
    s = s.str.strip()
    # Replace typical Hockey-Reference symbols and formatting
    s = s.str.replace(r"[\*\+]", "", regex=True)  # remove footnote markers * and +
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace("—", "-", regex=False)  # em dash to hyphen
    # Empty strings to NaN for cleaner numeric coercion
    s = s.replace({"": pd.NA, "nan": pd.NA})
    return s


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns to numeric where appropriate."""
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            cleaned = _clean_string_series(df[col])
            coerced = pd.to_numeric(cleaned, errors="coerce")
            # Heuristic: if at least 50% of non-null values convert to numbers, keep numeric
            non_null = cleaned.notna().sum()
            numeric_non_null = coerced.notna().sum()
            if non_null > 0 and numeric_non_null / max(1, non_null) >= 0.5:
                df[col] = coerced
            else:
                # Keep cleaned strings for textual columns
                df[col] = cleaned
    return df


def drop_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated header rows that scrape into the body (common in Hockey-Reference tables)."""
    for rk_col in ["Rk", "Rk_Rk", "Rk.1"]:
        if rk_col in df.columns:
            mask = ~(df[rk_col].astype(str).str.lower() == "rk")
            df = df.loc[mask]
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a sequence of basic cleaning steps."""
    df = flatten_columns(df)
    # Drop fully empty rows
    df = df.dropna(how="all")
    # Remove repeated header rows
    df = drop_header_rows(df)
    # Strip whitespace from column names again after filtering
    df.columns = [c.strip() for c in df.columns]
    # Coerce numerics where it makes sense
    df = coerce_numeric_columns(df)
    # Reset index after filtering
    df = df.reset_index(drop=True)
    return df


def fetch_table_by_id(url: str, table_id: str) -> Optional[pd.DataFrame]:
    """Fetch a Hockey-Reference table by HTML id with robust fallbacks that don't require lxml.

    Strategy:
    1) Try pandas.read_html directly by id.
    2) requests.get the page and parse with BeautifulSoup using best-available parser:
       prefer 'lxml', else gracefully fallback to 'html.parser'.
       - Look for wrapper div id=f"all_{table_id}" and extract the commented table.
       - If wrapper missing, scan all HTML comments to find one containing the table id.
    3) If BeautifulSoup parsing fails, use a regex-based extraction from HTML comments.
    4) Parse the extracted fragment with pandas.read_html without forcing a specific flavor.
    """
    # 1) Try direct read_html by table id
    try:
        tables = pd.read_html(url, attrs={"id": table_id}, flavor="bs4")
        if tables:
            return tables[0]
    except Exception as e:
        print(f"Direct read_html failed for id='{table_id}' from {url}: {e}")

    # Helper to parse tables from html fragment using pandas
    def _parse_tables_from_html(html_text: str) -> Optional[pd.DataFrame]:
        if not html_text:
            return None
        try:
            tables_local = pd.read_html(html_text, attrs={"id": table_id})
            if tables_local:
                return tables_local[0]
        except Exception:
            pass
        try:
            tables_local = pd.read_html(html_text, attrs={"id": table_id}, flavor="bs4")
            if tables_local:
                return tables_local[0]
        except Exception:
            pass
        return None

    html_text = None

    # 2) Fallback: fetch page and attempt BeautifulSoup-based extraction
    try:
        resp = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        })
        resp.raise_for_status()
        raw_html = resp.text

        # Try lxml first, fall back to html.parser if lxml is unavailable
        soup = None
        try:
            soup = BeautifulSoup(raw_html, "lxml")
            parser_used = "lxml"
        except Exception:
            soup = BeautifulSoup(raw_html, "html.parser")
            parser_used = "html.parser"

        # Prefer wrapper if available
        wrapper_id = f"all_{table_id}"
        wrapper = soup.find(id=wrapper_id)

        commented_html = None
        if wrapper:
            for c in wrapper.children:
                if isinstance(c, Comment):
                    commented_html = str(c)
                    break
            if not commented_html:
                table_tag = wrapper.find("table", id=table_id)
                if table_tag:
                    commented_html = str(table_tag)
        else:
            # Scan all comments if wrapper isn't found
            for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
                txt = str(c)
                if f'id="{table_id}"' in txt or f"id='{table_id}'" in txt:
                    commented_html = txt
                    break

        if commented_html:
            parsed = _parse_tables_from_html(commented_html)
            if parsed is not None:
                print(f"Parsed table '{table_id}' from commented HTML using {parser_used}.")
                return parsed

        # Keep raw_html for regex fallback
        html_text = raw_html
    except Exception as e:
        print(f"BeautifulSoup fallback failed for id='{table_id}' from {url}: {e}")

    # 3) Regex-based fallback: find HTML comments and extract the one containing the table id
    try:
        if not html_text:
            # If we don't have the raw HTML yet, try to get it now
            resp2 = requests.get(url, timeout=20)
            resp2.raise_for_status()
            html_text = resp2.text
        # Find all HTML comment blocks
        for m in re.finditer(r"<!--(.*?)-->", html_text, flags=re.S):
            block = m.group(1)
            if (f'id="{table_id}"' in block) or (f"id='{table_id}'" in block):
                parsed = _parse_tables_from_html(block)
                if parsed is not None:
                    print(f"Parsed table '{table_id}' from regex-extracted commented HTML.")
                    return parsed
    except Exception as e:
        print(f"Regex fallback failed for id='{table_id}' from {url}: {e}")

    print(f"All strategies failed to fetch table id='{table_id}' from {url}.")
    return None


def save_to_csv(df: pd.DataFrame, base_filename: str, downloads_dir: Optional[str] = None) -> str:
    downloads = downloads_dir or get_downloads_dir()
    os.makedirs(downloads, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{base_filename}_{timestamp}.csv"
    path = os.path.join(downloads, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def _normalize_player_name(name: str) -> str:
    if name is None:
        return ""
    return str(name).strip().lower()


def load_player_registry(json_path: str) -> dict:
    """Load a player registry from a local JSON file and build a mapping of name->master_id.

    Supports several common JSON shapes, including nested structures:
    - {"players": [ {"name": "...", "master_id": 123}, ... ]}
    - [ {"name": "...", "master_id": 123}, ... ]
    - [ {"teamAbbrev": "...", "players": [ {"name": "...", "master_id": 123}, ... ]}, ... ]
    - { "Some Player": {"master_id": 123}, ... }
    - { "Some Player": 123, ... }
    Recognizes name keys: name, player_name, player, full_name
    Recognizes id keys: master_id, masterId, id
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}

    def add_pair(nm, mid):
        if nm is None or mid is None:
            return
        try:
            mid_int = int(mid)
        except Exception:
            # ignore non-integer ids
            return
        mapping[_normalize_player_name(nm)] = mid_int

    def extract_from_obj(obj):
        if not isinstance(obj, dict):
            return False
        # Try to find name/id keys in a dict object
        name_keys = ["name", "player_name", "player", "full_name"]
        id_keys = ["master_id", "masterId", "id"]
        nm = None
        mid = None
        for k in name_keys:
            if k in obj and obj[k] not in (None, ""):
                nm = obj[k]
                break
        for k in id_keys:
            if k in obj and obj[k] not in (None, ""):
                mid = obj[k]
                break
        if nm is not None and mid is not None:
            add_pair(nm, mid)
            return True
        return False

    def walk(node):
        # Recursively traverse lists and dicts to find player dicts or nested 'players' arrays
        if isinstance(node, list):
            for it in node:
                walk(it)
        elif isinstance(node, dict):
            # If this dict itself is a player entry, try to extract
            found = extract_from_obj(node)
            # If it has a 'players' list, traverse it
            if "players" in node and isinstance(node.get("players"), list):
                walk(node["players"])
            # Also traverse all values to catch deeper nests
            for v in node.values():
                if isinstance(v, (list, dict)):
                    walk(v)

    # Start recursive walk from the root
    walk(data)

    return mapping


def map_master_ids_to_skaters(df: pd.DataFrame, registry_path: str, name_column: Optional[str] = None) -> pd.DataFrame:
    """Add a 'master_id' column by matching player names using the local registry.
    If no match is found, set 0 and print a console message listing unmatched names.
    """
    if df is None or df.empty:
        return df

    # Determine name column
    candidate_cols = [name_column] if name_column else []
    candidate_cols += ["Player", "Name", "player", "PLAYER"]
    name_col = next((c for c in candidate_cols if c and c in df.columns), None)

    # Heuristic fallbacks if not found by exact name
    if not name_col:
        # any column whose name contains 'player'
        contains_player = [c for c in df.columns if isinstance(c, str) and "player" in c.lower()]
        if contains_player:
            name_col = contains_player[0]
    if not name_col:
        # choose first likely name-like text column
        def is_name_like(series: pd.Series) -> bool:
            if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
                return False
            s = series.astype(str).dropna()
            if s.empty:
                return False
            sample = s.head(200)
            has_letters = sample.str.contains(r"[A-Za-z]", regex=True, na=False)
            has_spaces = sample.str.contains(r"\s", regex=True, na=False)
            is_not_numeric = ~sample.str.fullmatch(r"[-+]?\d+(\.\d+)?", na=False)
            score = (has_letters & is_not_numeric).mean() + (has_spaces & is_not_numeric).mean()
            return score / 2 >= 0.6
        for c in df.columns:
            try:
                if is_name_like(df[c]):
                    name_col = c
                    break
            except Exception:
                continue
    if not name_col:
        # cannot map without a name column
        print("Player name column not found in skaters dataframe; skipping master_id mapping. Available columns: " + ", ".join(map(str, df.columns)))
        df["master_id"] = 0
        return df

    print(f"Using '{name_col}' column for player name mapping.")

    # Load registry
    try:
        mapping = load_player_registry(registry_path)
    except FileNotFoundError:
        print(f"Player registry not found at: {registry_path}. Setting master_id=0 for all.")
        df["master_id"] = 0
        return df
    except Exception as e:
        print(f"Failed to load player registry from {registry_path}: {e}. Setting master_id=0 for all.")
        df["master_id"] = 0
        return df

    # Build normalized series and apply mapping
    norm_names = df[name_col].astype(str).map(_normalize_player_name)
    mapped = norm_names.map(mapping)

    # Fill missing with 0 and ensure integer
    master_id_series = pd.to_numeric(mapped, errors="coerce").fillna(0).astype(int)
    df["master_id"] = master_id_series

    # Report unmatched names
    unmatched = sorted(set(df.loc[df["master_id"] == 0, name_col].astype(str)))
    if unmatched:
        print(f"No master_id match for {len(unmatched)} player(s). Examples: {', '.join(unmatched[:20])}{' ...' if len(unmatched) > 20 else ''}")

    return df


def run():
    # 1) NHL 2026 skaters table
    url_skaters_2026 = "https://www.hockey-reference.com/leagues/NHL_2026_skaters.html#player_stats"
    table_id_skaters = "player_stats"
    skaters_df = fetch_table_by_id(url_skaters_2026, table_id_skaters)
    if skaters_df is not None:
        skaters_df = basic_clean(skaters_df)
        # Map player names to master_id using local registry
        registry_path = r"C:\Users\soluk\PycharmProjects\NHLfantasy\hr_players.json"
        skaters_df = map_master_ids_to_skaters(skaters_df, registry_path, name_column="Player")
        out1 = save_to_csv(skaters_df, "NHL_2026_skaters")
        print(f"Saved 2026 skaters to: {out1}")
    else:
        print("Skipping 2026 skaters export due to missing table.")

    # 2) NHL 2026 league team statistics table
    url_league_2026 = "https://www.hockey-reference.com/leagues/NHL_2026.html#all_stats"
    table_id_team_stats = "stats"  # Hockey-Reference id for 'Team Statistics'
    team_stats_df = fetch_table_by_id(url_league_2026, table_id_team_stats)
    if team_stats_df is not None:
        team_stats_df = basic_clean(team_stats_df)
        out2 = save_to_csv(team_stats_df, "NHL_2026_team_statistics")
        print(f"Saved 2026 team statistics to: {out2}")
    else:
        print("Skipping 2026 team statistics export due to missing table.")

    # 3) NHL 2026 advanced team statistics table
    url_league_2026_adv = "https://www.hockey-reference.com/leagues/NHL_2026.html#stats_adv"
    table_id_team_stats_adv = "stats_adv"  # Hockey-Reference id for 'Advanced Team Statistics'
    team_stats_adv_df = fetch_table_by_id(url_league_2026_adv, table_id_team_stats_adv)
    if team_stats_adv_df is not None:
        team_stats_adv_df = basic_clean(team_stats_adv_df)
        out3 = save_to_csv(team_stats_adv_df, "NHL_2026_team_statistics_advanced")
        print(f"Saved 2026 advanced team statistics to: {out3}")
    else:
        print("Skipping 2026 advanced team statistics export due to missing table.")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
            print("An unexpected error occurred:", exc)
            sys.exit(1)
