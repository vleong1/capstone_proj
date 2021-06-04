# Lover Recommendation with Unsupervised Learning

## Problem Statement

**Is it possible to predict the level of damage sustained by a building in the 2015 Gorkha earthquake in Nepal using classification modeling against the building’s features?**


## Executive Summary

## Table of Contents
- Background
- Data Collection
- Data Inspection/Cleaning
- NLP/Feature Engineering
- Data Compilation
- Data Visualization
- Imported Libraries
- Data Dictionary
- Data Modeling
- Analysis
- Conclusions and Recommendations
- Limitations and Future Project Refinements

## Background
On April 25, 2015, an earthquake measuring 7.8 on the Richter scale struck a region of the Asian nation of Nepal less than 50 miles northwest of the capital Kathmandu. Hundreds of aftershocks followed in the ensuing weeks, including a 7.3 magnitude quake 18 days later and two others measuring 6.5 and 6.6 on the Richter scale. About 8,900 people were killed and some 22,000 others injured by the temblor, which also damaged or destroyed over a million homes. Following the disaster, Nepal completed out a comprehensive household survey to assess building damage in the quake-affected districts. The main goal was to identify beneficiaries eligible for government housing reconstruction assistance. [link]('https://www.mercycorps.org/blog/quick-facts-nepal-earthquake#:~:text=Strength%3A%207.8%20on%20the%20Richter,strength%20and%20the%20resulting%20damage')

Earlier this year, the data science competition site DrivenData.org opened a contest to the public in an effort to discover the best classification model for predicting building damage in the Nepal earthquake. As of this project date, over 4,000 entrants had submitted models to the website.

### Data Collection

We began our experiment by downloading the datasets from the DrivenData.org [competition page]('https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/') containing information about hundreds of thousands of buildings that were examined in the earthquake zone. This consisted of a training set of data (with more than 260,000 observations), a test set of data (consisting of almost 89,000 observations), and a training label data (which had a target variable value associated with each observation in the training data).

The training and test datasets each contained 38 columns of data which included seven integer variables, eight categorical variables, and 22 binary variables. The target variable in the training and train label datasets was a three-value categorical variable representing the level of damage sustained by each structure.

### Data Inspection/Cleaning

After checking the datasets for null values (and finding none), we visually inspected the data tables to get a sense of the information we were preparing to model. We noticed that several categorical variables consisted of single letter values which had no relevance to any of the data points. These values were intentionally obfuscated by the competition organizers so as to reduce the amount of information provided to the entrants.

Because of these lettered values, we proceeded to label encode the values in the training and test datasets so that the data would be in numerical form, which would allow us to model these categories along with the other features. Once encoding was complete, we then subsetted the training dataset by pulling out the fisrt ten percent of observations. This subsetted dataset was the one we used for modeling purposes in an effort to save computer runtime during the machine learning process.

### Data Sources 
- [OkCupid Profiles (Kaggle)]('https://www.kaggle.com/andrewmvd/okcupid-profiles')

### Data Visualization



### Imported Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import sys
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity
from category_encoders import OneHotEncoder

```

### Data Dictionary

|Feature|Data Type|Possible Values|Description|
|---|---|---|---|
|**age**|*integer*|*18 - 69*|Age of person.|
|**status**|*category*|*single, available*|Relationship status of person.|
|**sex**|*category*|*m, f*|Gender of person.|
|**orientation**|*category*|*straight, gay, bisexual*|Relationship status of person.|
|**body_type**|*category*|*average, fit, athletic, rather not say, etc.*|Gender of person.|
|**diet**|*category*|*anything, vegetarian, vegan, kosher, halal, other*|Gender of person.|
|**drinks**|*category*|*never, sometimes, often*|Gender of person.|
|**sex**|*category*|*m, f*|Gender of person.|
|**sex**|*category*|*m, f*|Gender of person.|
|**sex**|*category*|*m, f*|Gender of person.|
|**sex**|*category*|*m, f*|Gender of person.|
|**sex**|*category*|*m, f*|Gender of person.|
|**sex**|*category*|*m, f*|Gender of person.|

### Data Modeling

Parameters used:
+ Logistic Regression: Penalty['l1', 'l2', 'elasticnet', 'none'], C[.01,.1,1,10,100]
+ Naive Bayes: Alpha[.01,.1,1,10,100]

## Analysis


### Conclusions/Recommendations



### Limitations and Future Project Refinements