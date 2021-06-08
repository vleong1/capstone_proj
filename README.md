# Lover Recommendation with Unsupervised Learning

## Problem Statement
Using unsupervised learning, can we build a model to recommend potential matches for a given user who's looking for a partner?

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

### Background


### Data Collection



### Data Inspection/Cleaning

Intensive cleaning, null values, grouping values, etc...

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

*Note: Some possible values of features have been truncated within the dictionary to show as an example.*

|Feature|Data Type|Possible Values|Description|
|---|---|---|---|
|**age**|*integer*|*18 - 69*|Age of person.|
|**status**|*category*|*single, available*|Relationship status of person.|
|**sex**|*category*|*m, f*|Gender identity of person.|
|**orientation**|*category*|*straight, gay, bisexual*|Sexual orientation of person.|
|**body_type**|*category*|*average, fit, athletic, rather not say, etc.*|Body type of person.|
|**diet**|*category*|*anything, vegetarian, vegan, kosher, halal, other*|Type of diet the person follows and how strictly it's followed. Possible values have been truncated to show as example.|
|**drinks**|*category*|*often, desperately, socially, rarely, not at all*|Alcohol consumption habits of person.|
|**drugs**|*category*|*yes, sometimes, never*|If the person partakes in drugs.|
|**pets**|*category*|*likes/dislikes dogs/cats*|Whether the person likes cats, dogs, or neither. Possible values have been truncated to show as example.|
|**offspring**|*category*|*doesn't have/has kid(s), doesn't want/wants kid(s)*|Whether or not the person has kids or plans on having them. Possible values have been truncated to show as example.|
|**smokes**|*category*|*yes, no, sometimes, when drinking, trying to quit*|Smoking habits of person.|
|**religion**|*category*|*atheism, christianity, catholicism, [religion] and serious about it, [religion] and laughing about it, etc.*|Religious beliefs of person and how strictly it's followed. Possible values have been truncated to show as example.|

### Data Modeling

Parameters used: (if applicable)


### Analysis


### Conclusions/Recommendations



### Limitations and Future Project Refinements
- Majority of dataset was from California (NorCal)
- Null values are ambiguous
- Some values, in general, are ambiguous