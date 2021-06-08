import streamlit as st
import pickle
import numpy as np
import pandas as pd

# load model
from recommender_model import lover_recommender_test

# load data
cupid_df = pd.read_pickle('../data/grouped_cupid.pkl')
cupid_religion = pd.read_pickle('../data/religion_cupid.pkl')

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://wallpaperaccess.com/full/3198513.jpg");
# background-size: cover;
# }
# </style>
# '''

st.set_page_config(layout = "wide", page_title = 'Lover Recommender App', page_icon = '💘')

st.title("💘 Lover Recommender App 💘")
st.header("Looking For A Partner? 💑👩‍❤️‍👩👨‍❤️‍👨")
st.markdown("----------------------------------------")


# user input features
st.subheader("Tell us a little bit about yourself first...")

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
religion_choice = st.radio('Does religion matter to you?',['yes', "no"])
if religion_choice == 'yes':
    religion = st.selectbox("Please specify your religion.", ('agnosticism', 'christianity', 'catholicism', 'judaism', 'buddhism', 'hinduism', 'islam', 'other'))
else:
    religion = "doesn't matter"

# put together all answers from user input
invalue = np.array([age, sex, orientation, body_type, diet, drinks, drugs, offspring, pets, religion, smokes])

st.subheader("Show Me My Top 5 Matches!!!")
click = st.button("Click Here")

if click:
    st.markdown('*Hint*: The higher the similarity, the **stronger** the match!')
    st.table(lover_recommender_test(invalue, religion, lowest_age, highest_age))