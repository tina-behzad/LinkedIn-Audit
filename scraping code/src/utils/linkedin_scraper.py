import traceback
import random

import chromedriver_autoinstaller
import pyotp
from httpcore import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup as bs
import re as re
from datetime import datetime
import time
import pandas as pd

from src.database import Database


class LinkedInScraper:

    def __init__(self, username, password, db_connection):
        chromedriver_autoinstaller.install()
        self.db_connection = db_connection
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.linkedin.com/login")

        self.driver.find_element(By.ID, "username").send_keys(username)
        password_ = self.driver.find_element(By.ID, "password")
        password_.send_keys(password)
        password_.send_keys(Keys.RETURN)
        time.sleep(random.uniform(2, 4))

    def search(self, query):
        search_bar = self.driver.find_element(By.CSS_SELECTOR, "input[placeholder='Search']")
        search_bar.send_keys(query)
        search_bar.send_keys(Keys.RETURN)
        time.sleep(2)
        try:
            # Wait for the filter section to load, then find the Jobs filter specifically
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="search-reusables__filters-bar"]/ul/li[0]/button'))).click()
        except Exception as e:
            print("Failed to apply the Jobs filter:", e)

        time.sleep(2)
        results = []

        jobs_block = self.driver.find_elements(By.CLASS_NAME,'jobs-search-results-list')
        # if len(jobs_block) == 0:
        #     dropdown_button = self.driver.find_element(By.XPATH,
        #                                           "//button[@id='navigational-filter_resultType']")
        #     dropdown_button.click()
        #     time.sleep(1)
        #     desired_option = self.driver.find_element(By.XPATH,
        #                                          "//li[text()='Jobs']")
        #     desired_option.click()
        #     # select = Select(self.driver.find_element(By.ID,'navigational-filter_resultType'))
        #     # select.select_by_visible_text('Jobs')
        #     # jobs_block = self.driver.find_elements(By.CLASS_NAME, 'jobs-search-results-list')
        jobs_list = jobs_block[0].find_elements(By.CSS_SELECTOR, ".jobs-search-results__list-item")
        # jobs = self.driver.find_elements(By.CSS_SELECTOR, ".job-card-container__link")
        links= []
        for job in jobs_list:
            all_links = job.find_elements(By.TAG_NAME,'a')
            for a in all_links:
                if str(a.get_attribute('href')).startswith("https://www.linkedin.com/jobs/view") and a.get_attribute(
                        'href') not in links:
                    links.append(a.get_attribute('href'))
                else:
                    pass

        job_titles = []
        company_names = []
        company_locations = []
        # work_methods = []
        post_dates = []
        # work_times = []
        job_desc = []

        for i in range(len(links)):
            try:
                self.driver.get(links[i])
                # i = i + 1
                time.sleep(2)
                # Click See more.
                self.driver.find_elements(By.CLASS_NAME,"artdeco-card__actions")[0].click()
                time.sleep(2)
            except:
                pass

            # Find the general information of the job offer
            # Used try, except in case there is some missing information for some of the job offers
            content = self.driver.find_elements(By.CLASS_NAME,'p5')[0]
            # for content in contents:
            try:
                job_titles.append(content.find_elements(By.TAG_NAME,"h1")[0].text)
                company_names.append(content.find_elements(By.CLASS_NAME,"job-details-jobs-unified-top-card__company-name")[0].text)
                informations = content.find_elements(By.CLASS_NAME,"job-details-jobs-unified-top-card__primary-description-container")[0].find_elements(By.CLASS_NAME,"tvm__text")
                company_locations.append(informations[0].text)
                # work_methods.append(informations[2].text)
                post_dates.append(informations[2].text)
                # work_times.append(content.find_elements(By.CLASS_NAME,"jobs-unified-top-card__job-insight")[0].text)
                # j += 1

            except:
                print("unable to get job details")
                time.sleep(2)

                # Scraping the job description
            # job_description = self.driver.find_elements(By.CLASS_NAME,'jobs-description__content')
            # for description in job_description:
            #     job_text = description.find_elements(By.CLASS_NAME,"jobs-box__html-content")[0].text
            #     job_desc.append(job_text)
            #     # print(f'Scraping the Job Offer {j}')
            #     time.sleep(2)

        df = pd.DataFrame(list(zip(job_titles, company_names,
                                   company_locations,
                                   post_dates)),
                          columns=['title', 'company_name', 'company_location', 'post_date'])
        # Storing the data to csv file
        # df.to_csv("./data/scraping_data/linkedin_job_search_results_{}.csv".format(query), index=False)
        # Output job descriptions to txt file
        # with open('job_descriptions.txt', 'w', encoding="utf-8") as f:
        #     for line in job_desc:
        #         f.write(line)
        #         f.write("\n")
        return df

    def open_recruiter(self):
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//span[@title='Recruiter']"))).find_element(By.XPATH,
                                                                                                   "./ancestor::a").click()
        # switch to the recruiter tab
        self.driver.switch_to.window(self.driver.window_handles[1])
        time.sleep(random.uniform(5, 8))
    def two_step_verification(self, secret_key):
        totp = pyotp.TOTP(secret_key)
        current_code = totp.now()
        self.driver.find_element(By.XPATH, "//input[@aria-label='Please enter the code here']").send_keys(current_code)
        self.driver.find_element(By.CSS_SELECTOR, "label[for='recognizedDevice']").click()
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))).click()
        time.sleep(random.uniform(2, 4))

    def recruiter_search(self,query,results_page_number =8, location = None):
        query_id = self.db_connection.get_query_id(query)
        self.recruiter_keyword_search(query)
        if location:
            self.filter_location(location)
        already_seen_names = set()
        for page_number in range(results_page_number):
            try:
                self.scrape_candidates_on_page(query_id, query, page_number, already_seen_names)
                self.go_to_next_page(page_number)
            except Exception as e:
                print("No more results or error:", e)
                break
                # self.scrape_candidates_on_page(query_id, query, page_number, already_seen_names)
                # self.go_to_next_page(page_number)
                # break

    def recruiter_keyword_search(self, query):
        search_bar = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Start a new search…']"))
        )
        search_bar.send_keys(query)
        search_bar.send_keys(Keys.RETURN)
        time.sleep(random.uniform(8, 10))

    def go_to_next_page(self, page_number):
        self.driver.execute_script("window.scrollTo(0, 0);")
        next_page_title = 'Go to next page {}'.format(page_number + 2)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"//a[@title='{next_page_title}']"))).click()
        time.sleep(random.uniform(5, 7))

    def get_candidate_information(self,card,query_id):
        info = WebDriverWait(card, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "lockup"))).text.split("\n", maxsplit=3)
        candidate_info = {}
        candidate_info['name'] = info[0]
        candidate_info['degree'] = info[2].split('·')[1]
        candidate_info['headline'] = info[3]
        candidate_info['is_enhanced'] = "Enhanced by resume" in card.text

        candidate_id = self.db_connection.get_candidate_id(candidate_info['name'], query_id)
        if not candidate_id:    #candidate doesn't exist
            candidate_info['img_url'] = self.get_candidate_image_url(card,high_quality_img = False)
            candidate_info['background'] = self.get_candidate_backgrounds(card)
            time.sleep(random.uniform(1, 2))
            candidate_info['skills'] = self.get_candidate_skills(card)
            candidate_info['recruiter_url'] = card.find_element(By.TAG_NAME,'a').get_attribute("href")
            return candidate_info

        else:
            candidate_info['candidate_id'] = candidate_id
            return candidate_info


    def get_candidate_backgrounds(self, card_element):
        background_dict = {}
        try:
            backgrounds = WebDriverWait(card_element, 5).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "history-group")))
        except:
            backgrounds = card_element.find_elements(By.CLASS_NAME,"history-group")
        if backgrounds:
            for background in backgrounds:
                buttons = background.find_elements(By.XPATH, ".//button[span[contains(text(), 'Show all')]]")
                # click on show all if it exists
                if buttons:
                    try:
                        WebDriverWait(card_element, 5).until(EC.element_to_be_clickable(buttons[0])).click()
                    except:
                        try:
                            self.driver.execute_script("window.scrollBy(0, -50);")
                            WebDriverWait(card_element, 5).until(
                                EC.element_to_be_clickable(buttons[0])).click()
                        except:
                            pass
                    # buttons[0].click()

                all_items_elements = background.find_element(By.CLASS_NAME, "expandable-list").find_elements(By.TAG_NAME, "li")
                all_items = [experience.text for experience in all_items_elements]
                if "Experience" in background.text:
                    background_dict["experience"] = all_items
                if "Education" in background.text:
                    background_dict["education"] = all_items
        else:
            print("no backgrounds found")
        if not "experience" in background_dict:
            background_dict["experience"] = None
        if not "education" in background_dict:
            background_dict["education"] = None
        return background_dict


    def get_candidate_skills(self, card_element):
        skills_elements = card_element.find_elements(By.CLASS_NAME, "decorations")
        if not skills_elements:  # If no decorations class is found, return None
            return None

        skills_element = skills_elements[0]  # Get the first matching element

        # Try to find and click the "Show all" button if it exists
        show_more_buttons = skills_element.find_elements(By.XPATH, ".//button[contains(., 'Show all')]")
        if show_more_buttons:
            try:
                WebDriverWait(card_element, 5).until(EC.element_to_be_clickable(show_more_buttons[0])).click()
            except:
                try:
                    self.driver.execute_script("window.scrollBy(0, -50);")
                    WebDriverWait(card_element, 5).until(EC.element_to_be_clickable(show_more_buttons[0])).click()
                except:
                    pass
        all_skills = card_element.find_elements(By.CLASS_NAME, "base-decoration")
        skills = [skill.text for skill in all_skills if skill.text.strip()]
        return skills if skills else None

    def get_candidate_image_url(self, card_element, high_quality_img = False):
        img_url = card_element.find_elements(By.TAG_NAME, "img")
        img_exists = (len(img_url) != 0) and img_url[0].get_attribute("src").startswith("https")
        #candidate has a profile picture, try getting a bigger image by clicking on the candidate
        if img_exists and high_quality_img:
            card_element.find_element(By.TAG_NAME, 'a').click()
            time.sleep(random.uniform(2, 5))
            bigger_img_url = self.driver.find_element(By.ID, "profile-container").find_element(By.TAG_NAME,
                                                                              "img").get_attribute("src")
            #close candidate profile window
            self.driver.find_element(By.CLASS_NAME, "pagination-header").find_element(By.XPATH,".//div/a").click()
            time.sleep(random.uniform(1, 2))
            if bigger_img_url:
                return bigger_img_url
            else:
                return img_url[0].get_attribute("src")
        elif img_exists and (not high_quality_img):
            return img_url[0].get_attribute("src")
        return None
    def quit(self):
        self.driver.quit()

    def filter_location(self, location):
        self.driver.find_element(By.XPATH, "//*[text()='Candidate geographic locations']").click()
        location_box = self.driver.find_element(By.CSS_SELECTOR, "input[placeholder='enter a location…']")
        location_box.send_keys(location)
        time.sleep(random.uniform(2, 4))
        location_box.send_keys(Keys.ARROW_DOWN)
        location_box.send_keys(Keys.RETURN)
        time.sleep(random.uniform(5, 8))

    def scrape_candidates_on_page(self,query_id, query, page_number, already_seen_names):
        candidates = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//ol[@class='ember-view profile-list']/li"))
        )

        for rank, card in enumerate(candidates):
            time.sleep(random.uniform(1, 3))
            self.driver.execute_script("window.scrollBy(0, 350);")
            try:
                candidate_info_dict = self.get_candidate_information(card, query_id)
                candidate_exists = "candidate_id" in candidate_info_dict
                if candidate_exists:
                    candidate_id = candidate_info_dict['candidate_id']
                else:
                    candidate_id = self.db_connection.insert_candidate(candidate_info_dict['name'],
                                                    query_id, candidate_info_dict['degree'],
                                                    candidate_info_dict['headline'],
                                                    candidate_info_dict['background'], candidate_info_dict['skills'],
                                                    candidate_info_dict['img_url'],
                                                    candidate_info_dict['recruiter_url'])

            except Exception as e:
                # traceback.print_exc()
                print("unable to get initial information for rank {} {}.".format(page_number * 25 + rank + 1, query))
                time.sleep(random.uniform(1, 2))
                continue

            name = candidate_info_dict['name']
            if name in already_seen_names:
                print("duplicate")
            already_seen_names.add(name)
            self.db_connection.insert_into_ranking( candidate_id, page_number * 25 + rank + 1,
                                candidate_info_dict['is_enhanced'])
            # results.append({'rank':page_number*25 + rank+1,'Name': name, 'Connection level': degree, 'headline': headline, 'Background': background_dict, 'img_src' :img_url})

