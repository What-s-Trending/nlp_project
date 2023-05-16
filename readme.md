# Github Trending Programming Language Analysis


# Project Description

Github is home to many different ideas and projects from all walks of life. We will attempt to identify drivers that can help predict the programming
language a Github Reposistory is using. We want to be able to idenfity the trending programming languages in an attempt to help new developers find 
indemand skills and good resources for learning since Github is open-source

# Project Goal

The goal of this project is to predict a repositories programming language. We plan to use only the readme text with any indication of the programming language being removed from the text. Our machine learning model will use theses parammeters to predict and will be evaluated for accuracy.

# Initial Hypotheses

- Is there a readme for each repository in the trending programming languages??
- Is there an equal breakdown in the trending programming languages?
- Does readme length play a factor in the programming language?
- Which language has the most of specific terms or concepts?
- Can we improve language prediction performance by using additional features or data sources?
- Are there any common words or phrases that are indicative of specific languages?


# Project Plan

- Planning - Steps required to replicate our process is presented in the readme file.

- Acquisition - 

- Preparation - 

- Exploration - We are looking to gain insights into the programming language distribution in GitHub repositories and to identify any patterns or trends related to programming languages based on readme text. This information can be useful for developers and organizations to prioritize their language learning or investment decisions.

- Modeling - We will go into modeling trying to predict the specific programming language of each repository using accuracy as a metric and only using text from the readme of each repository. No indication of the specific programming language is in the text. We will be using four predicitive models and evaluating the accuracy of each.

- Delivery - We will be packaging our findings in a final report python notebook. Results will be posted on GitHub.


# Data Dictionary

| Field 		   |        Data type 		|				Description				       |
|------------------|------------------------|----------------------------------------------|
| repo             |                  object| Unique GitHub repository   				   |
| language         |                  object| Computer programming language                |
| orignal          |                  object| The original readme text from the repo	   |
| clean            |                  object| The cleaned text from each repo readme       |
| stemmed          |                  object| Stemmed version of text from each repo readme|
| lemmatized       |                  object| Lemmatized version of text from each readme  |
| readme_length    |                   int64| Word count of each repo readme			   |
| language_code    |                    int8| Programming language id                      |


# Steps to Reproduce

To recreate our findings, you will need the nlp_project.ipynb file along with all .py files from the GitHub repository stored in the same directory on your device. The dataset file is provided in the repository as well.
 