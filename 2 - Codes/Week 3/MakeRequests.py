# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 23:47:05 2023

https://www.kyoshin.bosai.go.jp/kyoshin/instant/index_en.html

@author: Ahmed_A_Torky
"""

import requests

#Let's request the tall buildings information
response = requests.get("https://data.sfgov.org/resource/5kya-mfst.json")

#Let's check the response is OK (return code 200)
if(response.status_code == 200):
    #We will convert the response to json 
    tallBuildings = response.json()
    
    #we will read the first building in the response and print some information
    building = tallBuildings[0]
    print("Building Name", ["name"])
    print("\tOccupancy: ", building["occupancy"])
    print("\tAddress: ", building["address"])
    

import sys
import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup

#Starting the browser and opening tax assessor's data website for Anchorage
browser = webdriver.Chrome() 
url = "https://www.muni.org/pw/public.html"
browser.get(url)


#Fill parcel search box with zero
parcelBox1 = browser.find_element_by_name("PAR1")
parcelBox1.send_keys('0')

#Click on Submit
submitButton = browser.find_element_by_name("submitbtn") 
submitButton.click()

for i in range(1,3):
    print("Processing results page {}".format(i))
    #Finding the parcels table with XPath
    parcelsTable = browser.find_elements_by_xpath("//table[1]/tbody/tr[2]/td/table[5]/tbody/tr")
    parcelsCount = len(parcelsTable)

    #Looping through the parcels table rows
    for i in range(1, parcelsCount):
        #extracting the parcel id
        parcelRow = browser.find_element_by_xpath("//table[1]/tbody/tr[2]/td/table[5]/tbody/tr[{}]".format(i+1))
        parcelIdCell = parcelRow.find_element_by_xpath(".//td[1]")
        if parcelIdCell is not None:
            parcelId = parcelIdCell.text
            
            #Clinking on the parcel to open the parcel details
            parcelIdCell.click()
            WebDriverWait(browser, 10)

            occupancy = 'unknown'
            #We can now parse the parcel details
            #e.g. let's find the occupancy
            try:
                parcelSoup = BeautifulSoup(browser.page_source, "html.parser")
                
                #We will assume the header is always the seventh table row (maybe using trial and error)
                parcelHeader = parcelSoup.findAll('tr')[7].text

                #We will read the header
                if('Residential' in parcelHeader):
                    occupancy = 'Residential'
                elif('Commercial' in parcelHeader):
                    occupancy = 'Commercial'

            except:
                "Error: Failed to process parcel: " + parcelId

            print("Parcel: {} is {}".format(parcelId, occupancy))

            #clicking on the back button
            backButton = browser.find_element_by_name("backbtn")
            backButton.click()
    #Going to next page in the search results       
    nextButton = browser.find_element_by_name("nextbtn")
    nextButton.click()

