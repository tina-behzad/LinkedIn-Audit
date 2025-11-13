import sqlite3

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz


class Database:
    def __init__(self, db_location):
        self.conn = sqlite3.connect(db_location)


    def get_query_name(self, query_id):
        """
        Fetches the query name for a given query_id from the 'queries' table.
        """
        # This SQL targets the 'query' column in the 'queries' table
        # based on the 'id' column.
        sql_query = """
            SELECT
                q.query
            FROM queries q
            WHERE q.id = ?
        """
        # The query_id passed here (e.g., 15) will be substituted
        # safely for the '?' placeholder.
        df = pd.read_sql_query(sql_query, self.conn, params=(query_id,))

        if df.empty:
            print(f"No query found for id: {query_id}")
            return None
        else:
            # This extracts the value from the 'query' column
            # of the first (and only) row returned.
            return df.loc[0, 'query']


    def get_ranking_data_for_particular_batch(self, query_batch_number):
        query = f"""
            SELECT 
            r.rank,
            r.candidate_id,
            c.full_name,
            c.connection_level,
            c.headline,
            c.experience,
            c.education,
            c.img_src_url,
            q.query,
            pd.first_name,
            pd.last_name,
            pd.gender,
            pd.race,
            pd.highest_education AS highest_education_level,
            pd.age,
            pd.gender_tina,
            pd.gender_sid,
            r.is_enhanced,
            r.date
        FROM rankings r
        JOIN candidates c ON r.candidate_id = c.id
        JOIN queries q ON c.query_id = q.id
        LEFT JOIN personal_details pd ON c.id = pd.candidate_id
        WHERE q.query_batch_number = {query_batch_number}
        """

        df = pd.read_sql_query(query, self.conn)
        return df


    def get_ranking_data_for_query(self, query_id):
        query = f"""
            SELECT 
            r.rank,
            c.full_name,
            c.connection_level,
            c.headline,
            c.experience,
            c.education,
            c.img_src_url,
            q.query,
            pd.first_name,
            pd.last_name,
            pd.gender,
            pd.race,
            pd.highest_education AS highest_education_level,
            pd.age,
            pd.gender_tina,
            pd.gender_sid,
            pd.highest_education,
            r.is_enhanced,
            r.date
        FROM rankings r
        JOIN candidates c ON r.candidate_id = c.id
        JOIN queries q ON c.query_id = q.id
        LEFT JOIN personal_details pd ON c.id = pd.candidate_id
        WHERE q.id = {query_id}
        """
        df = pd.read_sql_query(query, self.conn)
        return df


    def get_BLS_data(self):
        sql_query = """
                    SELECT *
                    FROM bls_data
                """
        df = pd.read_sql_query(sql_query, self.conn)
        return df

def pre_process_data(data, attribute=None):
    if attribute == 'binary_race':
        # Add a new binary_race column
        data['binary_race'] = data['race'].apply(lambda r: 'nh_white' if r == 'nh_white' else 'non_white')

    
    elif attribute == 'Gender' or attribute is None:
        # Rename 'gender' to 'Gender' column
        data.rename(columns={'gender': 'Gender'}, inplace=True)
        data['Gender'].fillna('Unknown', inplace=True)

    elif attribute == 'binary_headline':
        # Map headline to binary attribute
        def map_headline(h):
            try:
                fl = str(h)[0].upper()
                return "A-M" if ('A' <= fl <= 'M') else "N-Z"
            except:
                return None
        data['binary_headline'] = data['headline'].apply(map_headline)

    elif attribute == 'headline_contains_query':
        SIMILARITY_THRESHOLD = 90

        def check_row_similarity(row):
            headline = row['headline']
            query = row['query'] # Access the query column using the provided name

            # Handle NaN or non-string values in either column for the current row
            # Ensure comparison happens only if both are valid strings
            if pd.isna(headline) or not isinstance(headline, str) or \
               pd.isna(query) or not isinstance(query, str):
                return False # Cannot compare if either is invalid

            # Calculate similarity score (convert to strings just in case, though checked above)
            # .lower() for case-insensitivity
            score = fuzz.token_set_ratio(str(headline).lower(), str(query).lower())

            # Return boolean based on threshold
            return score >= SIMILARITY_THRESHOLD
        
        data['headline_contains_query'] = data.apply(lambda row: f'{check_row_similarity(row)}', axis=1)

    elif attribute == 'gender_tina' or attribute == 'gender_sid':
        # Do nothing
        print('pre-processing', attribute)
        print(data[attribute].value_counts())
        pass
    elif attribute =='binary_education':
        data['binary_education'] = data['highest_education_level'].apply(
            lambda r: 'unknown'
            if pd.isnull(r)
            else 'graduate' if r in ['Master', 'PhD']
            else 'Undergrad'
        )
    else:
        raise ValueError('Must apply a valid attribute.')

    return data


def add_LI_members_to_missing_ranks(data, group_attribute):
    """
    For each distinct (query, date) in `data`, find the missing ranks
    from 1..max_rank−1 and append one “LinkedInMember” row per missing rank,
    tagging it with group_attribute='LIM'. Works whether there is 1 or many queries.
    """
    def _add_lim_for_subset(df_subset):
        df = df_subset.copy()
        query = df['query'].iat[0]
        date  = df['date'].iat[0]
        max_rank = df['rank'].max()
        all_ranks = set(range(1, max_rank))
        missing_ranks = sorted(all_ranks - set(df['rank']))

        new_rows = []
        for idx, rank_val in enumerate(missing_ranks):
            row = {col: None for col in df.columns}
            row['rank']       = rank_val
            row['full_name']  = f'LinkedInMember {idx}'
            row[group_attribute] = 'LIM'
            row['first_name'] = 'LinkedInMember'
            row['last_name']  = str(idx)
            row['query']      = query
            row['date']       = date
            new_rows.append(row)

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df

    # ————————————————
    # Apply the above helper to each (query, date) slice:
    parts = []
    for (_, _), subset in data.groupby(['query', 'date']):
        parts.append(_add_lim_for_subset(subset))

    # Reassemble
    return pd.concat(parts, ignore_index=True)


def filter_data_by_day_num(data, day_number):
    # Filter the data for the specific day number
    dates = data['date'].unique()
    print(f"Unique dates in the dataset: {dates}")
    dates = sorted(dates)
    print(f"Sorted unique dates in the dataset: {dates}")
    
    if day_number < len(dates):
        date_to_filter = dates[day_number]
        data = data[data['date'] == date_to_filter]
        return data

    else:
        raise ValueError(f"Day number {day_number} exceeds the number of unique dates in the dataset.")


def get_number_of_days_for_query(db_path, query_id):
    """
    Return the total number of unique days for a query.
    """
    db_connection = Database(db_path)
    data = db_connection.get_ranking_data_for_query(query_id)

    data = pre_process_data(data)
    dates = data['date'].unique()

    return len(dates)


def load_query_batch_from_db(db_path, batch_id):
    """
    Load query batch from the database.
    """
    db_connection = Database(db_path)
    data = db_connection.get_ranking_data_for_particular_batch(batch_id)

    return pre_process_data(data)


def load_query_from_db(db_path, query_id, day_number=None, attribute=None, include_missing=False):
    """
    Load query from the database.
    """
    db_connection = Database(db_path)
    data = db_connection.get_ranking_data_for_query(query_id)
    data = pre_process_data(data, attribute)

    if day_number is not None:
        data = filter_data_by_day_num(data, day_number)

        # Include missing ranks as "linkedin members"
        if include_missing:
            data = add_LI_members_to_missing_ranks(data, attribute)

    return data


def get_query_name(db_path, query_id, truncation_length=20):
    """
    Get name of query with id query_id
    """
    db_connection = Database(db_path)
    data = db_connection.get_query_name(query_id)
    # Only return the first truncation_length characters
    if truncation_length:
        return data[:truncation_length]
    else:
        return data