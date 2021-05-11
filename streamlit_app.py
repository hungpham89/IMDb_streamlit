from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st
st.set_page_config(layout="wide")
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import joblib
import string, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from imdb import IMDb
nltk.download('stopwords')

st.image('IMDb_Header_Page.jpg', use_column_width = 'auto')
st.write('''
# IMDb Review Predictor 
Developed by [Hung Pham](https://www.linkedin.com/in/hungpham89/)  

Source code can be found [here](https://github.com/hungpham89/IMDb_streamlit)

This is the demo app for my Capstone project at Brainstation, Data Science Bootcamp.

The project is to demonstrate how to use Machine Learning, specifically Natural Language Processing (NLP) to deal with text data and predict the sentiment within them. 

The app will grab the first 25 reviews from the user's review page, along with their scores and make the predictions based on the review content.

The [model](https://github.com/hungpham89/IMDB_review_predictor) was trained from 50k reviews from 150 movies.

Search for your movie below and select the one that you're looking for. 

You can hover over the reviews to see the full text.

''')
#Instantiate imdb instance
ia = IMDb()
#Get user input
user_input = st.text_input("Search for your movie here:")
#Wait until user input some text
if user_input == '':
    pass
else:
    #Using imdbpy to seach for movies and return the first 10 results
    movies = ia.search_movie(user_input)[0:10]
    results = [i['long imdb title'] for i in movies]
    #Make radio choice for user to choose from the results list
    choice = st.radio(label = 'Search results:', options = results)
    #Grab the link to our movie
    try:
        url1 = ia.get_imdbURL(ia.search_movie(choice)[0])
        st.write('IMDb link to this movie:')
        st.write(url1)

        #setup user agent for BS4, except some rare case, it would be the same for most browser     
        user_agent = {'User-agent': 'Mozilla/5.0'}
        response1 = requests.get(url1, headers = user_agent)
        soup1 = BeautifulSoup(response1.text, 'html.parser')

        review_link = 'https://www.imdb.com'+soup1.find('a', text = 'USER REVIEWS').get('href')

        url = review_link
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
        try:
            review = pd.DataFrame(data = data)
            st.write(movie_name)
            review[['Review Rating','temp']] = review['Review Rating'].str.split(pat = '/', expand = True)
            review.drop('temp', axis=1, inplace=True)
            review['Review Rating'] = review['Review Rating'].astype(int)
            st.dataframe(review) #print out the review dataframe after created it
            cont = True
        except:
            st.write('This movie doesn\'t have any user review, please select another one!')
            cont = False




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


        def hit_or_miss(s):
            is_max = s == 'Hit'
            return ['background-color: mediumseagreen' if v else '' for v in is_max]

        if cont != True:
            pass
        else:
            try:
                st.write('''
                ## After getting the reviews, we can make some predictions.

                The model will read the review's body text and guess the score of this review based on the content.

                If the prediction fall within 1-2 score from the true value, it will be counted as a `Hit`, any further and it will be counted as a `Miss` 
                ''')
                clean_text_model = joblib.load('Joblib_files/clean_text.joblib')
                countvec_model = joblib.load('Joblib_files/col_tran.joblib')
                LogAT = joblib.load('Joblib_files/logAT.joblib')

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
                st.dataframe(review.style.apply(hit_or_miss, subset = ['Hit or Miss']))
            except:
                st.write('This movie doesn\'t have any user review, please select another one!')
    except:
        st.write("This movie is not available now, please try again later!")
