#Testing, implement into github
<<<<<<< HEAD
#test ver 2 
#Now we have a conflict, let's fix it

#test and learn how to fix git conflict
>>>>>>> branch_2
#import streamlit
import streamlit as st

# Import Selenium and its sub libraries
import selenium 
from selenium import webdriver
# Import BS4
import requests #needed to load the page for BS4
from bs4 import BeautifulSoup
import pandas as pd #Using panda to create our dataframe
import numpy as np
#Import necessary libraries for modeling
import joblib
pd.set_option('display.max_colwidth', None)
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
#Import tensorflow to load model
import tensorflow as tf
from tensorflow import keras

st.image('IMDB.png', width=200)
st.write('''
# Simple IMDB Reviews Scraping 

Quick and easy way to get your text data for NLP

''')



url1 = st.text_input('Please paste your movie\'s link')
if url1 == '': #only run the rest after user pasted a link to input box and hit enter
    pass
else:
    #Using the same script to grab review, the only difference is we only scrap the review from 1st page, so dont need Selenium to hit expand.
    user_agent = {'User-agent': 'Mozilla/5.0'}
    response1 = requests.get(url1, headers = user_agent)
    soup1 = BeautifulSoup(response1.text, 'html.parser')

    review_link = 'https://www.imdb.com'+soup1.find('a', text = 'USER REVIEWS').get('href')

    url = review_link


    #setup user agent for BS4, except some rare case, it would be the same for most browser 
    user_agent = {'User-agent': 'Mozilla/5.0'}
    #Use request.get to load the whole page
    response = requests.get(url, headers = user_agent)
    #Parse the request object to BS4 to transform it into html structure
    soup = BeautifulSoup(response.text, 'html.parser')

    block = soup.find_all('div', class_ = 'review-container')
    movie_name = soup.find('div', class_ = 'parent').text.strip()
    review_title = []
    content = []
    rating = []
    date = []
    user_name = []
    for i in range(len(block)):
        try:
            ftitle = block[i].find('a', class_ = 'title').text.strip()
            fuser = block[i].find('span', class_ = 'display-name-link').text.strip()
            fcontent = block[i].find('div', class_ = 'text show-more__control').text.strip()
            frating = block[i].find('span', class_ = 'rating-other-user-rating').text.strip()
            fdate = block[i].find('span', class_ = 'review-date').text.strip()
        except:
            continue
    
        review_title.append(ftitle)
        user_name.append(fuser)
        content.append(fcontent)
        rating.append(frating)
        date.append(fdate)

    #Build data dictionary for dataframe
    data = {'User_name': user_name,
        'Review date' : date, 
        'Review title': review_title,
        'Review_body' : content, 
        'Review Rating': rating,
        }
    #Build dataframe to export
    review = pd.DataFrame(data = data)
    st.write(movie_name)
    review[['Review Rating','temp']] = review['Review Rating'].str.split(pat = '/', expand = True)
    review.drop('temp', axis=1, inplace=True)
    review['Review Rating'] = review['Review Rating'].astype(int)

    st.write(review) #print out the review dataframe after created it



#Part 2 make some prediction
#Define a manual function to clean text data
class CleanText(BaseEstimator, TransformerMixin):
    '''
    Define class and method to clean text data to prepare for vectorizer.
    Reference: Copied from 'https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27'
    Note: There is some slight modification to fit my project
    '''
   
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X



#Set up buttone to click to start prediction part
if url1 == '': #Check if url is empty, if yes, do nothing
    pass
else: #only show this part if url has been filled
    st.write('''
## Okay, now let's make some prediction
Using the same data that we scraped above
''')
    result = st.button("Let's go")
    if result: #Process if user click on 'Let's go' button
        #Loading model to predict score
        #Please note that I used the file path as is in my laptop, please change it accordingly to where you put my file. In the submission, I included it in Data folder
        clean_text_model = joblib.load(r'C:\Users\ngoch\Dropbox\Data science\Brainstation\Capstone project\Final\clean_text.joblib')
        countvec_model = joblib.load(r'C:\Users\ngoch\Dropbox\Data science\Brainstation\Capstone project\Final\col_tran.joblib')
        LogAT = joblib.load(r'C:\Users\ngoch\Dropbox\Data science\Brainstation\Capstone project\Final\logAT.joblib')

        #Clean the text
        cleaned_text = clean_text_model.transform(review['Review_body']) #transfrom text using CleanText class defined above
        cleaned_text = pd.DataFrame(data = cleaned_text)
        #Transform with countvec (actually using Tfidf in my final model)
        counted = countvec_model.transform(cleaned_text)

        #Predict the score
        y_hat = LogAT.predict(counted)
        y_hat = y_hat.round().squeeze()
        # Convert extreme value into their respective min,max
        y_hat = np.where(y_hat >10,10,y_hat)
        y_hat = np.where(y_hat <1,1,y_hat)
        y_hat.astype(int)

        #Add the newly predicted values to review dataframe
        review['Predicted'] = y_hat
        #Create new column to check hit or miss base on 2steps adjacent accuracy
        review['Hit or Miss'] = np.where(abs(review['Predicted']-review['Review Rating'])<=2, 'Hit', 'Miss')
        review
    else:
        pass