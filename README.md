# Analytical-analysis-of-US-hate-crime-globally
Understanding people's attitude and emotion toward hate crime and racism in social media

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#import-libraries">Import Libraries</a>
    </li>
    <li><a href="#data-collection">Data Collection</a></li>
    <li><a href="#eda">EDA</a></li>
    <li><a href="#sentiment-analysis">Sentiment Analysis</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- OVERVIEW -->
## Overview
The world has been hit hard by a coronavirus pandemic starting 2020. Hate crime and racism in US accelerated during the corona virus pandemic since then. The origin of Covid pandemic is presumably attributed to Asia without substance. Additionally, racism incidents due to unfortunate death of George Floyd in 2020 has given rise to Black Lives Matter and other movements. This research we try to understand people’s attitude toward hate crime and racism in social media, namely twitter, during pandemic around the globe We compare attitudes from two Asian subcontinents and European region to North America during this time. Additionally, we study the underlying emotions when people use social media to express hate crime and racism concerns in these regions. We study the negative-ness of the overall sentiment and dig into their dominant emotions. Our analysis uses North America as a baseline, and controls for subjectivity as a moderator.


<!-- Import Libraries -->
## Import Libraries
To run the code, open the tweet_hate_speech_research.ipynb file. 

```
# data manipulation
import pandas as pd
import numpy as np

#plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# natural language processing
import nltk
from nltk.corpus import stopwords
import string
import plotly.express as px
import re
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
nltk.download('punkt')
from nltk.tokenize import word_tokenize 

#sentiment and emotion analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as t2e

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

<!-- DATA COLLECTION-->
## Data Collection

Data is collected from Twitter using the official Twitter API, connecting to Twitter's database and retrieving historical data. The data then is directly injected into a MongoDB database for storage and easier manipulation. The keywords used are: "hate crime", "Asian hate crime",  "#hatecrime", "racism", "#asianhatecrime", "#BLM", "#blacklivesmatter",  "#chinesevirus" and "#stopasianhate". Four regions are selected in this study: India, Asia, North America, and Europe. The data collection rate is once per week for a total of 3 weeks. The first collection is from Jan/15/2022 to Jan/23/2022; the second collection is from Jan/23/2022 to Jan/30/2022; the third collection is from Jan/30/2022 to Feb/06/2022. The cumulative three weeks data count for each region is 809 entries for India, 351 entries for Asia, 4861 entries for Europe, and 2330 entries for North America.

Note that the retweets are filtered to avoid amplifying the sentiment.

<!-- EDA -->
## EDA

<!-- sentiment-analysis-->
## Sentiment Analysis

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
[1] Croucher (2020). Prejudice toward Asian Americans in the COVID-19 pandemic: the effects of social me- dia use in the United States. Frontiers in Communication, 5:39–39, 2020.

