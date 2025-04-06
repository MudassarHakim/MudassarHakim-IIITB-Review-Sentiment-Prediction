# Sentiment Analysis using RNN with LSTM
> This project implements a sentiment analysis model using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) architecture. The model is trained on a dataset of reviews to predict the sentiment of the text as positive or negative.


## Table of Contents
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

## General Information
The goal is to develop a model that can accurately predict the sentiment of the reviews. The project uses a deep learning approach, specifically RNN with LSTM, to learn the patterns in the text data and make predictions.

## Features
- RNN with LSTM Architecture: The model uses a combination of RNN and LSTM to learn the sequential patterns in the text data.
- Sentiment Analysis: The model predicts the sentiment of the text as positive or negative.
- Text Preprocessing: The project includes text preprocessing steps such as tokenization, stopword removal, and stemming.

## Conclusions
The model achieves an train accuracy of 89% and test accuracy of 75%.

## Example of making predictions
review = "This movie was fantastic! I loved the plot and the acting was amazing."
print(f"Review Sentiment: {predict_sentiment(review)}")
Review Sentiment: Positive

review1 = "This movie could have been better! the plot seemed pale and flow was monotonous."
print(f"Review Sentiment: {predict_sentiment(review1)}")
Review Sentiment: Negative 

## Technologies Used
- Python 3.x: The project is implemented in Python 3.x.
- TensorFlow/Keras: The project uses TensorFlow/Keras for building and training the model.
- NumPy/Pandas: The project uses NumPy and Pandas for data manipulation and analysis.
- NLTK/Spacy: The project uses NLTK and Spacy for text preprocessing.

## Acknowledgements
- This project was created as a case study required for Executive PG Programme in Machine Learning & AI - IIIT, Bangalore

## Contact
- Linkedin:
    - [Mudassar Hakim](https://www.linkedin.com/in/mudassar-ahamer-hakim-281b8b9/)
