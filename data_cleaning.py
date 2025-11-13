import os
import traceback
from datetime import datetime,date
import re
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

from src.database import Database
from src.utils.data_utils import clean_name, query_gender_api, query_genderize_api, estimate_age, map_degree, \
    split_name, query_apis_for_gender, education_analysis
import pandas as pd
import ethnicolr as ec
from nameparser import HumanName


class LINKEDIN_ANALYZER():

    def __init__(self, database):
        """
        Args:
            database (Database class)
        """
        self.database = database

    def extract_personal_details(self):
        unprocessed_candidates = self.database.get_candidates_without_personal_details()
        if unprocessed_candidates.empty:
            print("no candidate needs further processing.")

        unprocessed_candidates['first_name'], unprocessed_candidates['last_name'] = split_name(unprocessed_candidates['full_name'].str.split(',').str[0])
        assigned_gender_data = self.assign_gender(unprocessed_candidates)
        assigned_gender_data[['highest_education', 'age']] = assigned_gender_data['education'].apply(
            lambda x: pd.Series(education_analysis(x)))
        assigned_race_data = ec.pred_fl_reg_name(assigned_gender_data, 'last_name', 'first_name')
        assigned_race_data.drop(columns=['asian','hispanic','nh_black','nh_white','full_name','education'], inplace=True)
        assigned_race_data.rename(columns={'id':'candidate_id'},inplace=True)
        self.database.add_personal_details(assigned_race_data)

    def assign_gender(self,unprocessed_candidates):
        already_queried_names = self.database.get_gender_for_names(pd.DataFrame(unprocessed_candidates['first_name']).rename(columns={'first_name':'name'}))
        print(f'number of unique names missing: {len(already_queried_names[already_queried_names["gender"].isna()]["name"].unique())}')
        names_without_gender = already_queried_names[already_queried_names['gender'].isna()]
        unprocessed_candidates['first_name'] = unprocessed_candidates['first_name'].str.lower()
        unprocessed_candidates = unprocessed_candidates.merge(already_queried_names.rename(columns={'name':'first_name'}).drop_duplicates(subset=['first_name']), how = 'left', on='first_name')
        # we might have queried names using both apis and recieved nothing, these names are stored in database with null gender (if and only if both apis have returned null)
        names_not_queried = self.database.get_missing_names(names_without_gender)
        names_to_query = names_without_gender.loc[names_without_gender['name'].isin(names_not_queried), 'name'].tolist()
        if names_to_query:
            new_gender_data,not_queried_names = self.query_api_for_data(names_to_query)
            #the limit for gender api was reached for queries so we will drop these columns and not process them further, have to run the analyzer again in 24 hours
            if not_queried_names:
                print("#############not all data was processed due to limit on gender api!#####################")
                unprocessed_candidates = unprocessed_candidates[~unprocessed_candidates['first_name'].isin(not_queried_names)]
            new_gender_data.rename(columns={'name': 'first_name'}, inplace=True)
            unprocessed_candidates = unprocessed_candidates.merge(new_gender_data, on='first_name', how='left', suffixes=('', '_api'))
            unprocessed_candidates['gender'] = unprocessed_candidates['gender'].combine_first(unprocessed_candidates['gender_api'])
            unprocessed_candidates.drop(columns=['gender_api'], inplace=True)
            unprocessed_candidates['gender'] = unprocessed_candidates['gender'].replace("null", pd.NA)
            print(f"total number of rows without gender is {unprocessed_candidates['gender'].isna().sum()} \n unique names without gender are {len(unprocessed_candidates[unprocessed_candidates['gender'].isna()]['full_name'].unique())}")
        return unprocessed_candidates

    def query_api_for_data(self, names_to_query):
        gender_data,names_not_queried = query_apis_for_gender(names_to_query)
        self.database.add_new_names_data(gender_data)
        return gender_data[['name','gender']], names_not_queried





if __name__ == '__main__':
    db_connection = Database("../data/sqlite/linkedin.db")
    analyzer = LINKEDIN_ANALYZER(db_connection)
    analyzer.extract_personal_details()
    db_connection.close_connection()





