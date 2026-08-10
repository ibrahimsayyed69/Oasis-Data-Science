Email Spam Detection Project
This project implements a Machine Learning pipeline to classify email messages as either Spam or Ham (not spam) using Python, Natural Language Processing (NLP), and a Multinomial Naive Bayes classifier.

Project Structure & Workflow
Data Loading:

Reads the email dataset (EmailCollection.csv) using pandas with tab separation (sep='\t') and handles character encoding (latin-1).

Exploratory Data Analysis:

Uses seaborn and matplotlib to visualize the distribution of spam vs. ham labels.

Text Preprocessing:

Cleans text data using regular expressions (re) to remove non-alphabetical characters.

Converts text to lowercase and splits messages into individual tokens.

Removes English stop words and applies stemming via NLTK's PorterStemmer.

Feature Extraction:

Converts text corpora into numerical token-count vectors using CountVectorizer (with a limit of 3,500 max features).

Encodes target labels into binary format using pandas.get_dummies().

Model Training & Evaluation:

Splits data into training and testing sets (80/20 ratio) using train_test_split.

Trains a MultinomialNB classifier and measures accuracy performance with accuracy_score.

Model Persistence:

Saves both the trained vectorizer (cv-transform.pkl) and model (model.pkl) using Python's pickle library for future inference.

Requirements
To run this project, make sure you have the following Python libraries installed:

Bash
pip install pandas seaborn matplotlib nltk scikit-learn
