import pandas as pd
import numpy as np
import streamlit as st

from scipy import sparse
import sys
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity
from category_encoders import OneHotEncoder


# load in data
cupid = pd.read_pickle('../data/clean_cupid.pkl')
cupid_df = pd.read_pickle('../data/grouped_cupid.pkl')
# cupid_religion = pd.read_pickle('../data/religion_cupid.pkl')

# drop "status", "location"
cupid.drop(columns = ['status', 'location'], inplace = True)

# function to ohe, create sparse matrices, and return the cosine similarity based on orientation
def invalue_to_similarity(invalue_df, orientation_df):
    """
    invalue_df: converted DataFrame of user inputs
    orientation_df: DataFrame of all people of that orientation
    """
    
    # concat input values to orientation df to prep for cosine similarity
    df = pd.concat([orientation_df, invalue_df])

    # ohe
    df_encoded = OneHotEncoder(use_cat_names = True).fit_transform(df)
    
    # make cosine_similarity input (input X)
    cosine_input = pd.DataFrame(df_encoded.iloc[-1]).T
    
    # drop last encoded row (input Y -- data for input X to reference)
    df_encoded.drop(df_encoded.tail(1).index, inplace = True)
    
    # cosine_similarity(X, y)
    similarity = cosine_similarity(cosine_input, df_encoded)
    
    # return top 5 matches
    top5 = pd.DataFrame(similarity.tolist()[0], columns = ['similarity'], index = df_encoded.index).sort_values(by = 'similarity', ascending = False).iloc[:5]
    
    # return top 5 matches in a df with cosine similarities
    results = pd.DataFrame(columns = cupid.columns)

    for i in top5.index:
        results = results.append(pd.DataFrame(cupid.loc[i]).T)
    
    return results

# function to split the initial orientation_df by offspring sentiment
def offspring_subset(invalue_df, orientation_df):
    """
    invalue_df: converted DataFrame of user inputs
    orientation_df: DataFrame of all people of that orientation
    """
    
    # split by offspring preference
    if "doesn't have kid(s), but wants kid(s)" in invalue_df['offspring'].unique():
        orientation_df = orientation_df[(orientation_df['offspring'] == "has kid(s)") | (orientation_df['offspring'] == "has kid(s) and wants more") | \
                                        (orientation_df['offspring'] == "has kid(s), but doesn't want more")]
    elif "doesn't have kids, and doesn't want any" in invalue_df['offspring'].unique():
        orientation_df = orientation_df[(orientation_df['offspring'] == "doesn't have kids") | (orientation_df['offspring'] == "doesn't want kids") | \
                                        (orientation_df['offspring'] == "doesn't have kids, and doesn't want any")]
    elif "has kid(s), but doesn't want more" in invalue_df['offspring'].unique():
        orientation_df = orientation_df[(orientation_df['offspring'] == "doesn't have kids") | (orientation_df['offspring'] == "doesn't have kid(s), but wants kid(s)") | \
                                        (orientation_df['offspring'] == "wants kid(s)")]
    elif "has kid(s) and wants more" in invalue_df['offspring'].unique():
        orientation_df = orientation_df[(orientation_df['offspring'] == "has kid(s)") | (orientation_df['offspring'] == "has kid(s) and wants more") | \
                                        (orientation_df['offspring'] == "wants kid(s)")]
    
    # cosine similarities
    return invalue_to_similarity(invalue_df, orientation_df)

# recommender function -- v.8
def lover_recommender_test(invalue, religion, lowest_age, highest_age):
    """
    invalue (list): survey/streamlit app responses
    religion: religion of user
    lowest_age / highest_age: age range preference of partner recommendation
    """
   
    # convert input from array to DataFrame
    invalue_df = pd.DataFrame(invalue).T.rename(columns = {i:j for i,j in zip(np.arange(11), cupid_df.columns)})

    # ----------------
    
    # straight female looking for straight mmale
    if invalue_df['orientation'].unique()[0] == 'straight' and invalue_df['sex'].unique()[0] == 'f':
        
        # straight male df
        straight_male = cupid_df[(cupid_df['sex'] == 'm') & (cupid_df['orientation'] == 'straight') & (cupid_df['religion'] == religion) & \
            (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)].head(10000)
        
        # call 'invalue_to_similarity' function to return top 5 similarities
        try: 
            return offspring_subset(invalue_df, straight_male)
        except ValueError:
            return "Apologies, we don't have any profiles in the database that relate to your age range, orientation, and religious inputs at the moment."
        # return matches
        
        # see matches + option to see additional matches
        # return offspring_subset(matches, religion, lowest_age, highest_age)
    
    # straight male looking for straight female
    elif invalue_df['orientation'].unique()[0] == 'straight' and invalue_df['sex'].unique()[0] == 'm':
        
        # straight female df
        straight_female = cupid_df[(cupid_df['sex'] == 'f') & (cupid_df['orientation'] == 'straight') & (cupid_df['religion'] == religion) & \
            (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)].head(10000)

        # call 'invalue_to_similarity' function to return similarities
        try: 
            return offspring_subset(invalue_df, straight_female)
        except ValueError:
            return "Apologies, we don't have any profiles in the database that relate to your age range, orientation, and religious inputs at the moment."
        # return matches
        
        # see matches + option to see additional matches
        # return more_matches(matches, religion, lowest_age, highest_age)
    
    # gay male looking for gay male
    elif invalue_df['orientation'].unique()[0] == 'gay' and invalue_df['sex'].unique()[0] == 'm':
        
        # gay male df
        gay_male = cupid_df[(cupid_df['sex'] == 'm') & (cupid_df['orientation'] == 'gay') & (cupid_df['religion'] == religion) & \
            (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)]
        
        # call 'invalue_to_similarity' function to return similarities
        try: 
            return offspring_subset(invalue_df, gay_male)
        except ValueError:
            return "Apologies, we don't have any profiles in the database that relate to your age range, orientation, and religious inputs at the moment."
        # return matches
        
        # see matches + option to see additional matches
        # return more_matches(matches, religion, lowest_age, highest_age)
    
    # gay female looking for gay female
    elif invalue_df['orientation'].unique()[0] == 'gay' and invalue_df['sex'].unique()[0] == 'f':
        
        # gay female df
        gay_female = cupid_df[(cupid_df['sex'] == 'f') & (cupid_df['orientation'] == 'gay') & (cupid_df['religion'] == religion) & \
            (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)]
        
        # call 'invalue_to_similarity' function to return similarities
        try: 
            return offspring_subset(invalue_df, gay_female)
        except ValueError:
            return "Apologies, we don't have any profiles in the database that relate to your age range, orientation, and religious inputs at the moment."
        # return matches
        
        # see matches + option to see additional matches
        # return more_matches(matches, religion, lowest_age, highest_age)
    
    # bisexual male/female looking for bisexual male/female
    elif (invalue_df['orientation'].unique()[0] == 'bisexual' and invalue_df['sex'].unique()[0] == 'f') or \
         (invalue_df['orientation'].unique()[0] == 'bisexual' and invalue_df['sex'].unique()[0] == 'm'):
        
        # bi individual df
        bi = cupid_df[(cupid_df['orientation'] == 'bisexual') & (cupid_df['religion'] == religion) & \
                (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)]
        
        # call 'invalue_to_similarity' function to return similarities
        try: 
            return offspring_subset(invalue_df, bi)
        except ValueError:
            return "Apologies, we don't have any profiles in the database that relate to your age range, orientation, and religious inputs at the moment."
        # return matches
        
        # see matches + option to see additional matches
        # return more_matches(matches, religion, lowest_age, highest_age)