import streamlit as st
import pickle
import numpy as np
import pandas as pd

# load model
from recommender_model import lover_recommender_test, invalue_to_similarity, offspring_subset

# load data
cupid_df = pd.read_pickle('../data/grouped_cupid.pkl')
cupid = pd.read_pickle('../data/clean_cupid.pkl')

# drop "status", "location"
cupid.drop(columns = ['status', 'location'], inplace = True)

st.set_page_config(layout = "wide", page_title = 'Lover Recommender App', page_icon = '💘')

# sidebar

st.title("💘 Lover Recommender App 💘")
st.header("Looking For A Partner? 💑👩‍❤️‍👩👨‍❤️‍👨")
st.markdown("----------------------------------------")


# user input features
st.subheader("Tell us a little bit about yourself first...")

# user_form = st.form(key = "my_form")

# input
age = st.number_input('How old are you?')

st.write("What's your age range?")
lowest_age = st.number_input('Lowest Age:')
highest_age = st.number_input('Highest Age:')

sex = st.radio('What gender do you identify as?',['m', 'f'])
orientation = st.radio('What sexual orientation do you identify with?',['straight', 'gay', 'bisexual'])
body_type = st.selectbox('What body type do you have?',('average', 'fit', 'thin', 'full figured', 'used up', 'rather not say'))
diet = st.selectbox('What does your diet consist of?',('anything', 'vegetarian', 'vegan', 'kosher', 'halal', 'other'))
drinks = st.radio('Do you consume alcoholic beverages?',['yes','no', 'sometimes'])
drugs = st.radio('Do you use drugs?',['yes', 'no', 'sometimes'])

# offspring
# do you have any children? yes/no
any_children = st.radio('Do you have any kid(s)/children?', ['yes', 'no'])

# do you want any [more] children? yes/no
more_children = st.radio('Do you want any [more] kid(s)/children?', ['yes', 'no'])

if any_children == 'no' and more_children == 'yes':
    offspring = "doesn't have kid(s), but wants kid(s)"
elif any_children == 'no' and more_children == 'no':
    offspring = "doesn't have kids, and doesn't want any"
elif any_children == 'yes' and more_children == 'no':
    offspring = "has kid(s), but doesn't want more"
elif any_children == 'yes' and more_children == 'yes':
    offspring = "has kid(s) and wants more"

pets = st.selectbox("What's your sentiment on dogs and/or cats?",('likes dogs', 'likes cats', 'likes dogs and cats', 'dislikes dogs and cats'))
smokes = st.radio('Do you smoke?',['yes', 'no', 'sometimes'])

# does religion matter?
religion = st.selectbox("Please specify your religion.", ("doesn't matter", 'agnosticism', 'christianity', 'catholicism', 'judaism', 'buddhism', 'hinduism', 'islam', 'other'))

invalue = np.array([age, sex, orientation, body_type, diet, drinks, drugs, offspring, pets, religion, smokes])

st.subheader("Show Me My Matches!!!")
click = st.checkbox("Check Here ✔")

if click:
    matches = lover_recommender_test(invalue, religion, lowest_age, highest_age)

    # show matches from OG cupid
    try: 
        st.table(matches)

        st.markdown("What profile(s) catch your eye? 👀")
        options = st.multiselect("", list(matches.index))

        # keep a running df of recommended matches
        all_recs = matches

        # keep a running df of profiles liked
        liked = pd.DataFrame(columns = cupid_df.columns)

        for value in options:
            liked = liked.append(cupid.loc[value])

        try: 
            # more matches
            st.markdown("Want to see more matches?")
            more = st.radio("", ['No', 'Yes'])

            if more == 'Yes':
                # df for function -- grouped cupid
                invalue_df = pd.DataFrame(columns = cupid_df.columns)
                
                for value in options:
                    invalue_df = invalue_df.append(cupid_df.loc[value])

                # straight female
                if invalue_df['sex'].unique()[0] == 'f' and invalue_df['orientation'].unique()[0] == 'straight':
                    straight_female = cupid_df[(cupid_df['sex'] == 'f') & (cupid_df['orientation'] == 'straight') & (cupid_df['religion'] == religion) & \
                        (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)].head(10000)

                    matches = offspring_subset(invalue_df, straight_female)

                # straight male
                elif invalue_df['sex'].unique()[0] == 'm' and invalue_df['orientation'].unique()[0] == 'straight':
                    straight_male = cupid_df[(cupid_df['sex'] == 'm') & (cupid_df['orientation'] == 'straight') & (cupid_df['religion'] == religion) & \
                        (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)].head(10000)

                    matches = offspring_subset(invalue_df, straight_male)

                # gay female
                elif invalue_df['sex'].unique()[0] == 'f' and invalue_df['orientation'].unique()[0] == 'gay':
                    gay_female = cupid_df[(cupid_df['sex'] == 'f') & (cupid_df['orientation'] == 'gay') & (cupid_df['religion'] == religion) & \
                        (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)]

                    matches = offspring_subset(invalue_df, gay_female)

                # gay male
                elif invalue_df['sex'].unique()[0] == 'm' and invalue_df['orientation'].unique()[0] == 'gay':
                    gay_male = cupid_df[(cupid_df['sex'] == 'm') & (cupid_df['orientation'] == 'gay') & (cupid_df['religion'] == religion) & \
                        (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)]

                    matches = offspring_subset(invalue_df, gay_male)

                # bi individual
                elif (invalue_df['orientation'].unique()[0] == 'bisexual' and invalue_df['sex'].unique()[0] == 'f') or \
                        (invalue_df['orientation'].unique()[0] == 'bisexual' and invalue_df['sex'].unique()[0] == 'm'):
                    bi = cupid_df[(cupid_df['orientation'] == 'bisexual') & (cupid_df['religion'] == religion) & \
                        (cupid_df['age'] >= lowest_age) & (cupid_df['age'] <= highest_age)]

                    matches = offspring_subset(invalue_df, bi)

                # duplicate recommendations
                #if 

                # to return OG cupid outputs, that were based off grouped values
                # df to show users -- OG cupid
                show_users = pd.DataFrame(columns = cupid.columns)
                
                for value2 in matches.index:
                    show_users = show_users.append(cupid.loc[value2])

                st.markdown("Here are more matches, based on the profile(s) that caught your eye earlier! 🤗")
                st.table(show_users)

                # any more profiles that users liked
                st.markdown("Any of these profiles catch your eye? 👁")
                options_more = st.multiselect("", list(show_users.index))

                for value in options_more:
                    liked = liked.append(cupid.loc[value])

                if options_more:
                    st.subheader("Here are all the profiles you liked:")
                    st.table(liked)
                    st.subheader("Go on and send a message! 💌 Happy chatting! 💞💓")
        except:
            st.markdown("**If you'd like to see more matches, please select profiles from above that piqued your interest! 🙎**")
        
        else:
            st.subheader("Here are all the profiles you liked:")
            st.table(liked)
            st.subheader("Go on and send a message! 💌 Happy chatting! 💞💓")
        
    except:
        st.image("https://media1.tenor.com/images/5a7baa3abccc024569143229fa700dd6/tenor.gif?itemid=10592594", width = 350)
        st.subheader("Apologies!! We don't have any profiles in the database that relate to your *age range*, *orientation*, *offspring sentiment* and *religious* inputs at the moment. 🙅😭")