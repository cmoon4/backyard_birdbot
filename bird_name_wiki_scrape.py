# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:14:46 2021

@author: cmoon
This script reads the scientific name of the bird species from the iNaturalist 
dataset and records the "common name" or the every day name for the specie.
"""

import wikipedia as wiki
import pandas as pd

# read the labelmap (downloaded from: https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1)
df = pd.read_csv('aiy_birds_V1_labelmap.csv')
# background is background
df.at[0,'common_name']='background'

# for all the other scientific names in the labelmap,
for index in range(1,len(df)):
    # search for the bird in Wikipedia, the first result is the common name.
    search_out=wiki.search(df.name[index],results=1)
    # amend the dataframe with the common name
    df.at[index,'common_name']=search_out[0]
    # just a progress update
    if index%10 == 0:
        print(index,'/',len(df))
    
# save the results as a .csv file.
df.to_csv('aiy_birds_V1_labelmap_amended.csv',index=False)