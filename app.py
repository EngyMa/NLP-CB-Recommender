import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import random
import flask
import joblib

data = pd.read_csv('data/IMDBdata_MainData_poster_refresh.csv')
data=data.fillna(' ')

cosine_sim = joblib.load('cs.pkl')



# Reset index of your main DataFrame and construct reverse mapping as before
data.set_index('title', inplace=True)
indices = pd.Series(data.index)
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    recommended_movies=[]
    idx = indices[indices==title].index[0]

   # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies except itself
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the names of the top 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(data.index)[i])

    return recommended_movies


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        title = flask.request.form['title']

        return flask.render_template('main.html', original_input={'title': title},
                                     result=get_recommendations(title))

if __name__ == '__main__':
    app.run(debug=True)

