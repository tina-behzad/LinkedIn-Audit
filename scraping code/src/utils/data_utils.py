import ast
import json
import os
import unicodedata
from datetime import datetime

import pandas as pd
import numpy as np
import requests
import re
from dotenv import load_dotenv, dotenv_values
from nameparser import HumanName

from src.utils.constants import bachelors, masters, phd, associate_degree, high_school

load_dotenv()
def normalize_unicode_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def clean_name(name):
    """
        Cleans a name by removing titles, parentheses, and extra spaces,
        then splits it into first and last names.

        Args:
            name (str): The raw name string to process.

        Returns:
            tuple: A tuple containing the first name and last name.
        """
    # Titles to be removed
    titles = ['Dr.', 'Ms.', 'Mss.', 'Sir', 'Mr.', 'Mrs.', 'Prof.', 'Miss', 'Ph.D.','phd']

    # Remove emojis
    cleaned_name = normalize_unicode_text(name)
    cleaned_name = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned_name).strip()
    cleaned_name = re.sub('|'.join(titles), '', cleaned_name)

    cleaned_name = cleaned_name.strip()
    parts = cleaned_name.split()
    if len(parts) > 2 and len(parts[0]) == 2 and parts[0].endswith('.'):  # Handle X. something
        return parts[1], ' '.join(parts[2:])
    if len(parts) > 1:
        splitted_name = parts[0].split('.')
        if len(splitted_name)>1 and len(splitted_name[0])==1: # Handle X.something
            return splitted_name[1], ' '.join(parts[1:])
        if len(splitted_name) > 1 and len(splitted_name[1]) == 0:   #handle something. X
            return splitted_name[0], ' '.join(parts[1:])

    return parts[0], ' '.join(parts[1:])

def query_gender_api(names, api_token):
    url = "https://api.genderapi.io/api/"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    payload = '&'.join([f'name={name}' for name in names]) + f"&key={api_token}"
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        response_data = response.json()
        if response_data.get("status", False):
            if len(names) == 1:
                return pd.DataFrame([response_data]).drop(columns=['status','used_credits','remaining_credits','expires','duration'])
            return pd.DataFrame(response_data["names"])
    else:
        print(response.status_code)
        print(response.text)
    return pd.DataFrame()

def query_genderize_api(names):
    url = "https://api.genderize.io"
    params = {}
    for idx, name in enumerate(names):
        params[f'name[{idx}]'] = name
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print(response.text)
    return pd.DataFrame()



def query_apis_for_gender(names):
    gender_api_token = os.getenv("GenderAPI_Token")
    gender_api_batch_size = 100
    genderize_api_batch_size = 10
    names_not_queried = []
    batches_1 = [names[i:i + gender_api_batch_size] for i in range(0, max(len(names),200), gender_api_batch_size)]
    if len(names) > 200:
        names_not_queried = names[200:]
    results_api_1 = pd.concat([query_gender_api(batch, gender_api_token) for batch in batches_1], ignore_index=True)
    results_api_1.rename(columns={"total_names": "count"}, inplace=True)
    if not results_api_1.empty:
        results_api_1.drop(columns=['q'], inplace=True)
        names_without_gender = results_api_1[results_api_1['gender'] == 'null']['name'].tolist()
        results_api_1 = results_api_1[results_api_1['gender'] != 'null']
    else:
        names_without_gender = names

    batches_2 = [names_without_gender[i:i + genderize_api_batch_size] for i in
                 range(0, max(len(names_without_gender),100), genderize_api_batch_size)]
    if len(names_without_gender) > 100:
        names_not_queried.append(names_without_gender[100:])
    results_api_2 = pd.concat([query_genderize_api(batch) for batch in batches_2], ignore_index=True)
    results_api_2['country'] = pd.NA
    final_results = pd.concat([results_api_1, results_api_2], ignore_index=True)
    final_results['gender'] = final_results['gender'].replace({'male': 'M', 'female': 'F'})
    final_results['gender'] = final_results['gender'].replace("null", pd.NA)
    final_results['gender'] = final_results['gender'].replace("None", pd.NA)
    return final_results, names_not_queried

def reformat_background_dict(background_dict):
    experience = background_dict["experience"]
    education = background_dict['education']
    if experience is None:
        experience_returned = None
    else:
        experience_returned = json.dumps(experience)
    if education is None:
        education_returned = None
    else:
        education_returned = json.dumps(education)
    return experience_returned, education_returned


def map_degree(education_list):
    for edu in education_list:
        if keyword_in_text(edu, phd):
            return "PhD"
    for edu in education_list:
        if keyword_in_text(edu, masters):
            return "Master"
    for edu in education_list:
        if keyword_in_text(edu, bachelors):
            return "Bachelor"
    for edu in education_list:
        if keyword_in_text(edu, associate_degree):
            return "Associate"
    for edu in education_list:
        if keyword_in_text(edu, high_school):
            return "High School"

    return None  # Return None if no match is found


def estimate_age(education_list):
    for edu in education_list:
        if keyword_in_text(edu, high_school):
            education_year = edu.split('·')
            if len(education_year) < 2:
                #highschool doesn't have year information, move to bachelors degree
                break
            else:
                years = education_year[1]
                if '–' in years:
                    highschool_end_year = int(years.split('–')[1])
                    current_year = datetime.now().year
                    estimated_age = current_year - highschool_end_year + 18
                    return int(estimated_age)
                elif years.isdigit():
                    highschool_end_year = int(years)
                    current_year = datetime.now().year
                    estimated_age = current_year - highschool_end_year + 18
                    return int(estimated_age)


    # If no high school match, check for bachelor's education
    for edu in education_list:
        if keyword_in_text(edu, bachelors):
            education_year = edu.split('·')
            if len(education_year) < 2:
                continue
            else:
                years = education_year[1]
                if '–' in years:
                    bachelor_start_year = int(years.split('–')[0])
                    current_year = datetime.now().year
                    estimated_age = current_year - bachelor_start_year + 18 #assuming start of undergraduate is at age 18
                    return int(estimated_age)
                elif years.isdigit(): #assuming single year represents end of bachelors
                    bachelor_end_year = int(years)
                    current_year = datetime.now().year
                    estimated_age = current_year - bachelor_end_year + 22  # assuming end of undergraduate is at age 22
                    return int(estimated_age)
    return None

def keyword_in_text(text, keywords):
    """Check if any keyword exists as a standalone word in the text."""
    for keyword in keywords:
        pattern = rf'(?<!\w){keyword}(?=\s|$|[.,()])'
        cleaned_text = text.split('·')[0]
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            return True
        pattern = rf'\b{re.escape(keyword)}(?:\b|[.,()])'
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            return True
    return False

def split_name(full_name_column):
    '''
    param full_name_column: a pandas Series of full names
    return: two pandas Series — first_name and last_name
    '''

    # Remove "Ir." or "Engr." from the beginning (case-insensitive)
    cleaned_names = full_name_column.apply(
        lambda x: re.sub(r'^(ir\.|engr\.)\s*', '', x.strip(), flags=re.IGNORECASE)
    )

    # Parse first and last names
    first_name = cleaned_names.apply(lambda x: HumanName(x).first)
    last_name = cleaned_names.apply(lambda x: f"{HumanName(x).middle} {HumanName(x).last}".strip())

    return first_name, last_name



def education_analysis(educations_list):
    if (educations_list != 'None') and (educations_list is not None):
        educations_list = ast.literal_eval(educations_list)
        age_estimate = estimate_age(educations_list)
        try:
            highest_level = map_degree(educations_list)
            return highest_level, age_estimate
        except:
            return None, age_estimate

    return None, None