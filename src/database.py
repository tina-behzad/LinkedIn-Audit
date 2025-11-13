import json
import sqlite3

import numpy as np
import pandas as pd

from src.utils.data_utils import reformat_background_dict


class Database:
    def __init__(self, db_location):
        self.conn = sqlite3.connect(db_location)
    def create_database(self):
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                location_filter TEXT,
                query_batch_number INTEGER,
                bls_query_id INTEGER NOT NULL,
                FOREIGN KEY (bls_query_id) REFERENCES bls_data(id) ON DELETE CASCADE
            )
        ''')

        # Create the profiles table with a foreign key to queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                full_name TEXT NOT NULL,
                connection_level TEXT,
                headline TEXT,
                experience TEXT,
                education TEXT,
                skills TEXT,
                img_src_url TEXT,
                recruiter_profile_url TEXT,
                FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
                UNIQUE (query_id, full_name)  -- Ensures uniqueness
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                date TEXT NOT NULL DEFAULT CURRENT_DATE,  -- Automatically assigns the current date
                rank INTEGER NOT NULL,
                is_enhanced INTEGER,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personal_details (
                candidate_id INTEGER PRIMARY KEY,  -- Also a Foreign Key
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                gender TEXT,
                race TEXT,
                age INTEGER,
                highest_education TEXT,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
            )
        ''')


        self.conn.commit()


    def move_bls_data(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bls_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Occupation TEXT NOT NULL,
                Total INTEGER,
                Gender_F REAL,
                race_nh_white REAL,
                race_nh_black REAL,
                race_asian REAL,
                race_hispanic REAL,
                age_16_19 REAL,
                age_19_24 REAL,
                age_24_34 REAL,
                age_34_44 REAL,
                age_44_54 REAL,
                age_54_64 REAL,
                age_64_plus REAL,
                Median_Age REAL,
                Gender_M REAL
            )
        ''')

        # Load data from CSV and insert into the occupations table
        df = (pd.read_csv("../data/gender_data/BLS_full_data.csv").drop(columns='Unnamed: 0')
              .rename(columns={'age_<19':'age_16_19', "age_19-24":"age_19_24","age_24-34":"age_24_34","age_34-44":"age_34_44","age_44-54":"age_44_54","age_54-64":"age_54_64","age_64+":"age_64_plus"}))
        df.to_sql("bls_data", self.conn, if_exists="append", index=False)

        self.conn.commit()


    def move_gender_data(self):
        ssa_data = pd.read_csv("../data/gender_data/SSA_2018.csv")
        ssa_data['Probability'] = pd.to_numeric(ssa_data['Probability'], errors='coerce')
        ssa_data = ssa_data.loc[ssa_data.groupby('Name')['Probability'].idxmax()].rename(columns={'Probability':'probability','Count':'count','Gender':'gender','Name':'name'})
        ssa_data['name'] = ssa_data['name'].str.lower()
        ssa_data['source'] = 'SSA'
        ssa_data['country'] = np.nan

        api_data = pd.read_csv("../data/gender_data/gender_api.csv")
        api_data['source'] = 'API'
        full_data = pd.concat([ssa_data,api_data])
        full_data = full_data.dropna(subset=['name'])
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS names_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                gender TEXT,
                count INTEGER,
                probability REAL,
                source TEXT,
                country TEXT
            )
        ''')
        full_data.to_sql("names_data", self.conn, if_exists="append", index=False)
        self.conn.commit()


    def fill_queries_data(self):
        cursor = self.conn.cursor()

        # Load data into the queries table from a Pandas DataFrame
        df_queries = pd.read_csv("../data/gender_data/CPS_employment_data.csv")  # Assuming queries data is stored in a CSV

        df_occupations = pd.read_sql("SELECT id,Occupation FROM bls_data", self.conn)

        # Insert data into queries table
        for _, row in df_queries.iterrows():
            # Find the best occupation match by longest name
            bls_matched_data = df_occupations[df_occupations["Occupation"].str.contains(row["title"], na=False, case=False)]
            if not bls_matched_data.empty:
                matched_row = bls_matched_data.loc[bls_matched_data["Occupation"].str.len().idxmax()]
                occupation_id = matched_row["id"]
            else:
                occupation_id = None

            cursor.execute('''
                INSERT INTO queries (query, location_filter, bls_query_id, query_batch_number)
                VALUES (?, ?, ?, ?)
            ''', (row["title"], "New York City Metropolitan Area", occupation_id, row["query_sequence"]))

        # Commit changes and close connection
        self.conn.commit()

    def get_query_batch(self,query_batch_number):
        cursor = self.conn.cursor()
        cursor.execute("SELECT query FROM queries WHERE query_batch_number = ?", (query_batch_number,))
        queries = [row[0] for row in cursor.fetchall()]  # Extracting the queries into a list
        return queries


    def get_query_id(self,query):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM queries WHERE query = ?", (query,))
        query_id = cursor.fetchone()[0]
        return query_id


    def close_connection(self):
        self.conn.close()

    def insert_candidate(self,name,query_id,connection_level,headline,background_dict,skills_list,img_url, recruiter_profile_url):
        experience, education = reformat_background_dict(background_dict)
        skills = json.dumps(skills_list) if skills_list else None
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO candidates (
                query_id, full_name, connection_level, headline, experience, 
                education, skills, img_src_url, recruiter_profile_url
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
        query_id, name, connection_level, headline, experience, education, skills, img_url, recruiter_profile_url))
        inserted_id = cursor.lastrowid
        self.conn.commit()
        return inserted_id

    def insert_into_ranking(self,candidate_id,rank,is_enhanced):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO rankings (candidate_id,rank, is_enhanced)
            VALUES (?, ?, ?)
                ''', (candidate_id, rank, is_enhanced))
        self.conn.commit()


    def get_candidates_without_personal_details(self):
        query = """
        SELECT c.id, c.full_name, c.education
        FROM candidates c
        LEFT JOIN personal_details p ON c.id = p.candidate_id
        WHERE p.candidate_id IS NULL;
        """

        # Execute the query and store the result in a Pandas DataFrame
        df = pd.read_sql_query(query, self.conn)
        return df

    def get_candidate_id(self,name, query_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM candidates WHERE full_name = ? AND query_id = ?", (name, query_id))
        result = cursor.fetchone()  # Fetch one row

        # Check if the candidate exists and get the ID
        if result:
            candidate_id = result[0]
        else:
            candidate_id = None
        return candidate_id

    def move_gender_label_data(self,csv_file, query_number, labeled_col_name):
        labeled_data = pd.read_csv(csv_file)
        labeled_data['candidate_id'] = labeled_data['full_name'].apply(
            lambda name: self.get_candidate_id(name, query_number)
        )
        col_name = 'gender_tina' if labeled_col_name == 'Tina' else 'gender_sid'
        labeled_data[labeled_col_name] = labeled_data[labeled_col_name].map({
            'male': 'M',
            'female': 'F'
        })
        cur = self.conn.cursor()

        sql = f"""
            UPDATE personal_details
               SET {col_name} = ?
             WHERE candidate_id = ?
            """

        # Build a list of tuples (new_value, key)
        params = [
            (row[labeled_col_name], row['candidate_id'])
            for _, row in labeled_data.iterrows()
        ]

        cur.executemany(sql, params)
        self.conn.commit()


    def assign_manual_label_data(self, manually_labeled_data_tuples,labeled_col_name):
        cur = self.conn.cursor()

        sql = f"""
                    UPDATE personal_details
                       SET {labeled_col_name} = ?
                     WHERE candidate_id = ?
                    """

        cur.executemany(sql, manually_labeled_data_tuples)
        self.conn.commit()

    def get_gender_for_names(self, names_df):
        gender_data = pd.read_sql_query("SELECT name, gender FROM names_data", self.conn)
        gender_data['name'] = gender_data['name'].str.lower()
        names_df['name'] = names_df['name'].str.lower()
        merged_df = names_df.merge(gender_data, on='name', how='left')

        return merged_df

    def get_missing_names(self, names):
        db_names = pd.read_sql_query("SELECT name FROM names_data", self.conn)
        db_names['name'] = db_names['name'].str.lower()
        names_df = pd.DataFrame(names, columns=['name'])
        names_df['name'] = names_df['name'].str.lower()
        missing_names_df = names_df.merge(db_names, on='name', how='left', indicator=True)
        missing_names = missing_names_df[missing_names_df['_merge'] == 'left_only']['name']
        return missing_names

    def add_new_names_data(self, names_data):
        names_data['source'] = 'API'
        names_data.to_sql("names_data", self.conn, if_exists="append", index=False, method="multi")

    def add_personal_details(self,personal_details_df):
        personal_details_df.to_sql("personal_details", self.conn, if_exists="append", index=False, method="multi", chunksize=100)


    def get_ranking_data_for_particular_batch(self, query_batch_number):
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
          pd.candidate_id ,
          c.full_name,
          c.img_src_url,
          pd.first_name,
          pd.last_name,
          pd.gender,
          pd.race,
          pd.gender_sid,
          pd.gender_tina
        FROM personal_details pd
        JOIN candidates      c  ON pd.candidate_id = c.id
        JOIN queries         q  ON c.query_id       = q.id
        WHERE q.id = {query_id};
        """
        df = pd.read_sql_query(query, self.conn)
        return df

    def get_complete_ranking_data(self):
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
            r.is_enhanced,
            r.date
        FROM rankings r
        JOIN candidates c ON r.candidate_id = c.id
        JOIN queries q ON c.query_id = q.id
        LEFT JOIN personal_details pd ON c.id = pd.candidate_id
        """
        df = pd.read_sql_query(query, self.conn)
        return df



