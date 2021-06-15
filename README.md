# Lover Recommendation with Unsupervised Learning - Veronica Leong


## Problem Statement

**Using unsupervised learning, can we build a model to recommend potential matches for a given user who's looking for a partner?**

## Table of Contents

- [Background](#Background)
- [Data Collection](#Data-Collection)
- [Data Inspection/Cleaning](#Data-Inspection/Cleaning)
- [Data Dictionary](#Data-Dictionary)
- [Data Visualization](#Data-Visualization)
- [Imported Libraries](#Imported-Libraries)
- [Modeling Approach](#Modeling-Approach)
- [Modeling Analyses](#Modeling-Analyses)
    - [Streamlit App]('https://share.streamlit.io/vleong1/lover-recommendations/main/app.py')
- [Conclusions/Limitations](#Conclusions/Limitations)
- [Recommendations](#Recommendations)
- [Future Project Refinements](#Future-Project-Refinements)

### Background
COVID has affected the entire world in many ways. One particular aspect that has been greatly affected is dating. Since the pandemic, users have turned to online dating to continue looking for romance. Match Group, who owns majority of the dating apps on the market (Tinder, Match.com, Hinge, OkCupid), has reported a [15% increase]('https://www.businessinsider.com/tinder-hinge-match-group-dating-apps-more-users-coronavirus-2020-8') in new subscribers in Q1 2020, right about the time stay-at-home orders went into place. Online dating became the new normal, as media outlets began suggesting virtual date ideas, such as virtual museum tours or ordering Uber Eats to share over FaceTime with a significant other. Dating apps were able to provide people with company, aiding in loneliness and isolation, as people were able to communicate and meet others. The apps also adapted to COVID guidelines, with Tinder having a pop up to remind users to remain socially distanced and to wash their hands.

### Data Source(s)

- [OkCupid Profiles (Kaggle)]('https://www.kaggle.com/andrewmvd/okcupid-profiles')

### Data Collection
I retrieved the OkCupid Profiles dataset from Kaggle, containing approximately 60k rows of OkCupid profile users, with their personal information stripped. The dataset was 14.3+ MB large and contained fields from age, sex, orientation, and short response text data with open ended questions.

### Data Inspection/Cleaning

I initially **dropped columns** I felt weren't indicative of a user's lifestyle, such as "Essay" (open ended responses), "Height", "Ethnicity", "Sign" (zodiac sign). Additionally, I **filtered out** for users whose "status" was *single* or *available*. Afterwards, I **filled null values** to the best of my ability. For example, if the "pets" feature was null, I'd assume a user didn't like dogs or cats. Lastly, I **grouped / generalized values**, since I noticed that if I were to model using the original, raw values, the most similar profiles returned would be exact matches or extremely similar to the user input, which may be beneficial if that's how one would like to have their recommendations returned. However, I wanted recommendations that were similar throughout the features, instead of 1:1.

**How null values were filled:**

|Feature|Value Filled|
|---|---|
|**body type**|rather not say|
|**diet**|anything|
|**drinks**|not at all|
|**drugs**|never|
|**offspring**|doesn't have kids|
|**pets**|dislikes dogs and cats|
|**religion**|doesn't matter|
|**smokes**|no|

**How values were grouped / generalized:**

<div><img src="assets/grouped_values.jpg" width="500"/></div>

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

### Data Visualization

<div><img src="assets/age_sex_distribution.jpg" width="900"/></div>
1. I first wanted to take a look at **age distribution by sex** and noticed that majority of the ages lied between the ages of 24 - 30, with predominately more male profiles across majority of the age groups. Interestingly, there are more female than male profiles for ages 55+.


<div><img src="assets/diet_religion_count.jpg" width="900"/></div>
2. I also wanted to take a look at **diet by religion**, where there were significantly less data points for halal or kosher diets, relative to vegetarian diets. Additionally, there was a lack of profiles of Islam religion, which majority of those data points also lived within a halal diet. Lack of data points, especially for specific religions, would affect the recommendations, since there would be a smaller pool of potential mathces and "religion" would be one of the matching criteria I'd specify that would determine recommendations for a user.


### Imported Libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from category_encoders import OneHotEncoder

```

### Modeling Approach

<div><img src="assets/content_based.jpg" width="300"/></div>

I built a recommender system model, specifically a **Content-Based Filtering** recommender model, since my project was based off a fixed dataset and isn't as robust as an actual dating app. This model works in the way where a user would indicate their interest in a profile(s) and based on similar profiles of the liked profile(s), the similar profiles would then be recommended back to the user. The **metric** I focused on for the project was *cosine similarity*, which would measure how similar 2 vectors are, regardless of their sizes, and return a value between 0 - 1. The higher the cosine similarity, the more alike the 2 objects/vectors that are being compared.

**Modeling Process:**
1. User would input their information, specifying age range, offspring sentiment, and if religion mattered to them
1. Overall dataset would be filtered based on the matching criteria noted above
1. User input/liked profile(s) and filtered dataset are encoded into 0/1 for the model to appropriately interpret and make predictions
1. Perform `cosine_simlarity` and return top matches based on matching criteria and similarity

**--> Content Based filtering -- If a user wanted to see additional profiles**
5. User indicates the profile(s) they liked
6. Repeat steps 2 - 4
7. Return additional matches of similar profiles based on the liked profile(s) back to the User

**Matching Criteria:**

<div><img src="assets/matching_criteria.jpg" width="500"/></div>

In terms of the **matching criteria**, there were *4 criteria* I wanted to ensure recommendations were filtered out for:
- **Orientation**: First and foremost, I think it's important to match a user based on their sex and orientation. For example, Straight Male matched with either a Straight Female or a Bisexual Female.
- **Age Range**: Everyone is in different stages of their life, depending on their age or where they are in life. Perhaps someone younger is looking for more mature individuals.
- **Religion**: Religion is one of, if not, the most important feature to recommend matches, since it portrays one's personal and cultural values. Simply, if religion *mattered* to a user, they'd be recommended profiles of users with the same religion. Contrastly, if religion *didn't matter* to a user, then they're be recommended profiles of users who are atheist or had a religion that wasn't serious to them.
- **Offspring Sentiment**: Lastly, I think it's important to match based off of a user's offspring sentiment, since to pursue and take a relationship to the next level, the topic of children normally come up. Simply, if a user *wanted kid(s)*, they'd be recommended profiles of users who have kid(s), and vice versa- if a user *didn't want kid(s)*, they'd be recommended profiles of users who didn't have kid(s).

### Modeling Analyses
Since there were no ratings included in the dataset to perform some sort of supervised learning that would evaluate my recommender models, my analysis mainly consisted of trying out different combinations of user inputs and evaluating the recommendations the model returned. Here are just a few user input combinations I thought had interesting findings:

#### **1. Gay Male**

**User Input:**
<div><img src="assets/gay_male_input.jpg" width="1100"/></div>

**Initial Matches (Liked profiles highlighted):**
<div><img src="assets/gay_male_matches.jpg" width="1100"/></div>

**Additional Matches (Based off liked profiles):**
<div><img src="assets/gay_male_more_matches.jpg" width="1100"/></div>

#### **--> Analysis: Gay Male**

**Initial Matches based on matching criteria + cosine similarity**
- Based on the matching criteria, the user input is that of a **gay male** with an **age range between 28 - 35**, who **has kid(s), but doesn't want more**, and religion **doesn't matter** for them. A user who's religion doesn't matter returns other users who are atheist or not serious/laughing about their religion. The initial recommendations returned gay males who all don't have kids.
- In terms of the other features that weren't part of the matching criteria, the user input has a diet consisting of `anything`, `yes` to drinks, and `no` to drugs, returning recommended profiles similar for the most part, aside from profile `42964` who indicated that they don't drink and profile `35743` who indicated that they dislike dogs and cats.

**Liked Profiles + Additional Matches**
- Profiles `16775`, `46102`, and `45626` were liked by the User, since across the board, the features were similar to the User's inputs.
- In terms of the **additional recommended profiles**, they are based off the liked profiles the User specified, while still performing cosine similarity against a filtered dataset based on the User's age range, orientation, and religion preference. The only matching criteria that is *directly related* to liked profile is `offspring sentiment`. Since all liked profiles are gay males who don't have kids, all additional profiles were also gay males who don't have kids.
- Additionally, there are still 2 additional matches that are slightly different from the User's profile/lifestyle -- profile `48214` that takes drugs `sometimes` and profile `46012` who doesn't drink, whereas the User indicated that they drink and don't take drugs.

#### **2. Bisexual Female**

**User Input:**
<div><img src="assets/bisexual_f_input.jpg" width="1100"/></div>

**Initial Matches (Liked profiles highlighted):**
<div><img src="assets/bisexual_f_matches.jpg" width="1100"/></div>

**Additional Matches (Based off liked profiles):**
<div><img src="assets/bisexual_f_more_matches.jpg" width="1100"/></div>

#### **--> Analysis: Bisexual Female**

**Initial Matches based on matching criteria + cosine similarity**
- Based on the matching criteria, the user input is that of a **straight female** with an **age range between 22 - 30**, who **doesn't have kids, and doesn't want any**, and **buddhism** religion.  A user specifying their religion indicates that religion matters to them, so the recommendations return other users who are Buddhist / serious about Buddhism as well. The initial recommendations returned a mix of straight males and bisexual females who all don't have kids.
- In terms of the other features that weren't part of the matching criteria, the user input has a diet consisting of `vegan`, `doesn't drink at all`, and `no` to drugs. Only profile `26260` indicated that they drank `socially`. Additionally, those of `vegan`, `vegetarian`, and `anything` diets were also recommended. In terms of pet sentiment, the User noted that she liked cats only, however profile `47399` indicated that the profile liked dogs only.

**Liked Profiles + Additional Matches**
- Although the User indicated that she smoked, the profiles she liked were `48349` and `24689`, since the pet sentiments most closely represents that of the user -- both profiles indicated that they have cats, whereas the User likes cats.
- In terms of the **additional recommended profiles**, they are based off the liked profiles the User specified, while still performing cosine similarity against a filtered dataset based on the User's age range, orientation, and religion preference. The only matching criteria that is *directly related* to liked profile is `offspring sentiment`. Since both liked profiles were straight males who didn't smoke, with a `fit` and `average` body type, all additional matches are straight males that have a `fit` or `athletic` body type, with all but one profile indicating that they didn't smoke.
- Additionally, because liked profile `48349` indicated that they like dogs and cats, the addiitonal matches have a mix of users who like both pets, dislike both pets, or just like dogs -- this feature doesn't match the pet sentiment of the user, but instead *moreso the liked profiles*.

### Streamlit App
Please feel free to test my [deployed Streamlit App]('https://share.streamlit.io/vleong1/lover-recommendations/main/app.py') for a hands-on lover recommendation experience!!

### Conclusions/Limitations
In conclusion, I noticed that the main limitation with *content-based filtering* is that **recommendations can become too similar**, due to the nature of the recommender model that has a limited ability to expand on predictions/recommendations, since the model is solely making predictions/recommendations based on the User's existing interests. The **dataset was also imbalanced**, lacking in the Islam religion or that of the Halal or Kosher diet. There was also significantly less users who had kids, versus those who didn't have kids. This would affect recommendations, especially if a user is of the Islam religion, with a Kosher diet, and wants kids. Additionally, the **more matching criteria** to filter the data on, the **smaller the pool** of users for recommendations. Lastly, I felt that some **values were ambiguous**, especially if left *null*. For example, I assumed that if "pets" was null, the user didn't like dogs or cats. However, it might be the case where they're unable to have a pet, not necessarily that they dislike pets.

### Recommendations
- To address an imbalanced dataset, my recommendation there would be to **gather more data**, especially for features that are lacking data points (diet, religion, offspring sentiment). 
- Additionally, I'd recommend to **group / generalize data values**, so that the model isn't looking for exact, 1:1 vector matches, but instead recommending profiles that have slight variation, after filtering the dataset based on the matching criteria
    - Ex) User said "yes" for drinks --> Model may recommend profiles of those who drink "often" or "desperately"
    - Ex) User said "likes dogs" for pets --> Model may recommend profiles of those who "likes/has dogs" or "likes/has dogs and dislikes cats" 

### Future Project Refinements
1. Add a sort of "rating" for each user profile **-->** This would assist in having a point of reference in determining if the recommended profiles for a user were a "good" match / how good of a match they were.
1. Handle profiles that were recommended again **-->** Depending on the combination of user inputs, there is a possibility of having a profile recommended again, if the user wanted to view additional recommendations based off the profile(s) they liked.
1. Incorporate additional relationship statuses (i.e. married, seeing someone) **-->** Perhaps one of my matching criteria could be if the user is open to a monogamous or polyamorous.

Updated: June 14, 2021