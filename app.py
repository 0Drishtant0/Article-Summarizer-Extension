import requests
from flask import Flask,render_template,url_for,jsonify
from flask import request as req
from flask_cors import CORS
from transformers import pipeline, set_seed
from newspaper import Article

from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
articles_df = joblib.load('articles_df.pkl')
vectorizer = joblib.load('vectorizer.pkl')
Amodel = joblib.load('article_model.joblib')



def process_text(text):
    """Process text function.
    Input:
        text: a string containing the text
    Output:
        processed_text: a list of words containing the processed text

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks    
    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    # tokenize text
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text_tokens = tokenizer.tokenize(text)

    processed_text = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # processed_text.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            processed_text.append(stem_word)

    return processed_text


## get recommendations
def find_related_articles(keywords):
    # Preprocess the user input keywords
    processed_keywords = process_text(keywords)
    processed_keywords = ' '.join(processed_keywords)
    
    # Transform the user input keywords into TF-IDF features
    keyword_features = vectorizer.transform([processed_keywords])
    
    # Find the nearest neighbors to the user input keywords
    distances, indices = Amodel.kneighbors(keyword_features)
    
    # Get the related articles based on the nearest neighbors
    related_articles = articles_df.iloc[indices[0]]
    
    return related_articles




nltk.download("punkt")

app = Flask(__name__, template_folder='template')
CORS(app)

@app.route("/", methods = ["GET", "POST"])

def Index():
  return(render_template('index.html'))
  

@app.route('/Summarise', methods = ['GET','POST'])

def Summarise():
   if req.method == 'POST':
        try:
            tokenizer = AutoTokenizer.from_pretrained("tokenizers")
            pipe = pipeline("summarization", model="pegasus-c3PO",tokenizer=tokenizer)
            maxL = int(req.json['maxL'])
            url = req.json['url']
            article = Article(url)
            article.download()
            article.parse()
            article_recommendations = find_related_articles(article.title)
            recommendations = [row.to_dict() for _, row in article_recommendations.iterrows()]
            urls = [movie['url'] for movie in recommendations]
            titles = [movie['title'] for movie in recommendations]
            print(urls)
            print(titles)
            gen_kwargs = {"length_penalty":maxL/550, "num_beams":40, "max_length": 180}
            result = pipe(article.text, **gen_kwargs)[0]["summary_text"].replace(" .<n>", ".\n")
            print(result)
            return jsonify({'summary_text':result, 'recommendations': urls, 'titles': titles})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

   else:
       return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.debug = True
    app.run()
