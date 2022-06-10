<h1 align="center">Topic Analysis</h1>

## Problem Description
The problem highlights the use of machine learning algorithms to categorize different comments scraped from an online platform and make relevant predictions about the topics
associated with those comments. There are a total of 40 topics to classify these comments. Even though the problem seems like a simple classification problem, as we dive deeper to
understand the data, we realize that the real problem asks us to make sense of the comments mentioned in the dataset and then assign categories.

We are working with 900,000 comments in total, which belonged to 40 categories, replaced with numerical labels. We can commence our EDA by thinking of techniques that would perform the initial task of converting the words in each comment into
their numerical representation. Understanding the distribution of the comments across the 40 categories is crucial too.

<img src="https://github.com/ShrutiL1396/Python/blob/main/Machine%20Learning%20Models/Topic%20Analysis/Dist.PNG" width="400" height="300">

## Initial Strategy and Outcome
The initial strategy we followed for tokenizing was to implement NLTK's word tokenizer, word_tokenize(). This is often popularly used to segregate words in a sentence into a list of
words, also known as 'tokens.' After cleaning up the tokens, removing stop words and unnecessary characters we performed word embedding using Word2Vec as it is one of the most popular 
embedding techniques that attempts to find semantic and syntactic similarities and relations with other words. When transforming the generated text into vectors using 
Term Frequency-Inverse Document Frequency (TFIDF) to evaluate how important a particular word is in the given collection of comments, we did encounter computational and memory issues.
Consequently, we utilized a subset of the 900,000 comments as our dataset to perform the initial training analysis. We observed that models such as Linear Support Vector Machines (SVM), 
which usually perform well on smaller datasets, seemed to perform poorly on our dataset since it consisted of several categories. Owing to this, we explored a transformer-based approach for training, such as BERT.

## Model Used
Pre-trained transformer based 'BERT Base Uncased' and 'BERT for Sequence Classification' for the text classification.

## Training set <br/>
900,000 comments 

## Test set  <br/>
55 comments

## Test Accuracy <br/>
50.84%  

## Contents 
- Code - [Topic Analysis](https://github.com/ShrutiL1396/Python/blob/main/Machine%20Learning%20Models/Topic%20Analysis/Topic_Analysis.ipynb) <br/>
- Test set - [Test set](https://github.com/ShrutiL1396/Python/blob/main/Machine%20Learning%20Models/Topic%20Analysis/TestFileTemplate.csv) <br/>
- Output file - [Output file](https://github.com/ShrutiL1396/Python/blob/main/Machine%20Learning%20Models/Topic%20Analysis/Output_File.csv) <br/>

## Contact
Shruti Shivaji Lanke - <br/>
shrutilanke13@gmail.com or slanke1@student.gsu.edu <br/>
