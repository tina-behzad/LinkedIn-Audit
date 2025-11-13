import random
import time
import pyotp
import os
from dotenv import load_dotenv, dotenv_values

from src.database import Database
from src.utils.linkedin_scraper import LinkedInScraper
import pandas as pd
load_dotenv()

if __name__ == '__main__':
    location = "X"
    db_connection = Database("./data/sqlite/linkedin.db")
    queries = db_connection.get_query_batch(8)
    scraper = LinkedInScraper(os.getenv("LINKEDIN_USERNAME"), os.getenv("LINKEDIN_PASS"), db_connection)
    scraper.two_step_verification(os.getenv("LINKEDIN_SECRET_KEY"))
    time.sleep(random.uniform(4, 8))
    scraper.open_recruiter()


    try:
        for index,query in enumerate(queries):
            scraper.recruiter_search(query, results_page_number=42)
            time.sleep(random.uniform(4, 8))
        scraper.quit()
        db_connection.close_connection()
    except:
        db_connection.close_connection()
        scraper.quit()
