import requests
from bs4 import BeautifulSoup
from selenium.webdriver import Firefox
import sys
import pandas as pd
import numpy as np

def soupify(html):
    return BeautifulSoup(html, "html.parser")

def get_cascade_header():
    '''
    This function will scrape cascadevacationrentals.com site to retrieve
    property_id and the url that accompanies it for further scraping of
    detail information.
    Rather than returning anything, this function will write its final product
    into a file under data directory.
    '''
    unit_names = []
    unit_url = []

    for i in range(1, 8, 1):
        response = requests.get("http://www.cascadevacationrentals.com/vacation-rentals-homes.asp?cat=7023&page={}&o=0".format(i))
        soup = soupify(response.text)

        properties = soup.find("div", {"class":"col-xs-12 col-sm-8 col-md-9",
                               "id":"content2Right"
                              }).find_all("div", {"class": "row-same-height row-full-height"})

        for unit in properties:
            unit_names.append(unit.find("li", {"class":
                                               "property-list-item property-list-code"
                                              }).find("strong").text)
            unit_url.append(unit.find("a")['href'])

    cascade_header_df = pd.DataFrame.from_items([('property_code', unit_names),
                                                 ('url', unit_url)])

    cascade_header_df.to_csv('data/cascade_header.csv')

    print('file successfully written to data/cascade_header.csv')


def scrape_cascade_details(df):
    # browser.find_elements_by_css_selector is ridiculous to type -- Miles Erickson
    select = browser.find_elements_by_css_selector
    select_one = browser.find_element_by_css_selector

    prop_details_columns = ['property_code', 'num_guests', 'num_bedrooms', 'num_bathrooms',
                            'allows_pets', 'property_size', 'manager_rating', 'property_rating']
    prop_details = []

    amenities_columns = ['property_code', 'amenities']
    amenities_list = []

    rates_table_columns = ['property_code','season', 'start', 'end', 'sun', 'mon', 'tue',
                           'wed', 'thu', 'fri', 'sat', 'min_nights']
    rates_table_list = []


    for i in range(len(df)):
        property_code = df.property_code.iloc[i]

        url = "http://www.cascadevacationrentals.com{}".format(df.url.iloc[i])
        browser.get(url)

        time.sleep(10)

        html = browser.page_source

        #Select the information from the first table
        # Number of Guests, Bedrooms, Bathrooms, Allows Pets, Property Size
        table_list = select_one('div.table-responsive > table.table.table-striped.table-bordered').text.split('\n')
        num_guests = table_list[0][7:]
        num_bedrooms = table_list[1][9:]
        num_bathrooms = table_list[2][10:]
        allows_pets = table_list[3][12:]
        if len(table_list) >= 5:
            property_size = table_list[4][14:]
        else:
            property_size = ''

        try:
            select_one('div.lr-info-block-property-rating-panel')
        except:
            manager_rating = 0
            property_rating = 0
        else:
            manager_rating = select_one('div.lr-info-block-property-rating-panel'
                               ).find_element_by_name('score').get_attribute('value')
            property_rating = select_one('div.lr-info-block-property-rating-property-panel'
                                ).find_element_by_name('score').get_attribute('value')

        prop_details.append([property_code, num_guests, num_bedrooms, num_bathrooms,
                             allows_pets, property_size, manager_rating, property_rating])

        #Select information from Amenities table
        amenities_raw = select_one('#content2Right > div:nth-child(7)')
        amen_1 = amenities_raw.find_elements_by_tag_name('ul')[0].text.split('\n')
        amen_2 = amenities_raw.find_elements_by_tag_name('ul')[1].text.split('\n')
        amenities_list.append([property_code, amen_1 + amen_2])

        rates_table = select_one('.responsive-rate-table > tbody:nth-child(1)'
                                ).text.split('\n')

        for i in range(1, len(rates_table), 1):
            row = rates_table[i].split(' ')

            try:
                int(row[-1])
            except:
                season = " ".join(rates_table[i].split()[:-9])
                start = row[-9]
                end = row[-8]
                sun = row[-7]
                mon = row[-6]
                tue = row[-5]
                wed = row[-4]
                thu = row[-3]
                fri = row[-2]
                sat = row[-1]
                min_nights = 1
            else:
                season = " ".join(rates_table[i].split()[:-10])
                start = row[-10]
                end = row[-9]
                sun = row[-8]
                mon = row[-7]
                tue = row[-6]
                wed = row[-5]
                thu = row[-4]
                fri = row[-3]
                sat = row[-2]
                min_nights = row[-1]

            rates_table_list.append([property_code, season, start, end, sun, mon, tue, wed,
                                     thu, fri, sat, min_nights])


        pd.DataFrame(prop_details, columns=prop_details_columns).to_csv('data/prop_head.csv')
        pd.DataFrame(amenities_list, columns=amenities_columns).to_csv('data/amenities.csv')
        pd.DataFrame(rates_table_list, columns=rates_table_columns)..to_csv('data/rates.csv')
