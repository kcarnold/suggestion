# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:05:11 2017

@author: kcarnold
"""
#%%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
#%%
import os
from boto.mturk.connection import MTurkConnection
mturk = MTurkConnection(aws_access_key_id=os.environ['aws_access_key_id'], aws_secret_access_key=os.environ['aws_secret_access_key'])
#%%
assignment, hit = mturk.get_assignment('3DR23U6WE5EZPZSPNKUZ6WN2FZ2ETB')
#%%
mturk.get_hit(hit.HITId)[0].RequesterAnnotation