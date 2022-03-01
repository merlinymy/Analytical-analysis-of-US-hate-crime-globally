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
    <li><a href="#data-cleaning">Data Cleaning</a></li>
    <li><a href="#eda">EDA</a></li>
      <ol>
        <li>
          <a href="#word-clouds">Word Clouds</a>
        </li>
        <li>
          <a href="#data-overview">Data overview</a>
        </li>
        <li>
          <a href="#top-frequent-words">Top Frequent Words</a>
        </li>
      </ol>
    <li><a href="#sentiment-analysis">Sentiment Analysis</a></li>
      <ol>
        <li>
          <a href="#sentiment-result">Sentiment Result</a>
        </li>
        <li>
          <a href="#polarity-result">Polarity Result</a>
        </li>
        <li>
          <a href="#subjectivity-result">Subjectivity Result</a>
        </li>
        <li>
          <a href="#emotion-result">Emotion Result</a>
        </li>
      </ol>
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

#### Regions data collected from
<img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/regions.png" width = "500" title = "regions data were collected">


<!-- data-cleaning -->
## Data Cleaning
We applied following natural language cleaning techniques: lemmatization and stemming, removed the stop words, omit any foreign characters (only kept ASCII characters), removed links from the tweets, replaced slang words.

Below is the function used for tweets cleaning.
```
def dfCleaning(df):

    #import pandas_profiling

    #df = df[['text','retweet_count','favorite_count']]
    df.drop_duplicates(inplace = True)
    #Lowercase
    df['clean_tweet']= df['text'].apply(lambda x : x.lower())
    #Code to remove https
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    #Code to remove @
    df['clean_tweet'] = df['clean_tweet'].apply(
        lambda x : ' '.join([tweet for tweet in x.split()if not tweet.startswith("@")]))
    #Removing numbers
    df['clean_tweet'] = df['clean_tweet'].apply(
        lambda x : ' '.join([tweet for tweet in x.split() if not tweet == '\d*']))
    #Removing all the greek characters using unidecode library
    df['clean_tweet'] = df['clean_tweet'].apply(
        lambda x : ' '.join([unidecode.unidecode(word) for word in x.split()])) 
    #Removing the word 'hmm' and it's variants
    df['clean_tweet'] = df['clean_tweet'].apply(
        lambda x : ' '.join([word for word in x.split() if not word == 'h(m)+' ]))
    #Code for removing slang words
    d = {'luv':'love','wud':'would','lyk':'like','wateva':'whatever','ttyl':'talk to you later',
                   'kul':'cool','fyn':'fine','omg':'oh my god!','fam':'family','bruh':'brother',
                   'cud':'could','fud':'food','gal':'girl'} ## Need a huge dictionary
    words = "I luv myself"
    words = words.split()
    reformed = [d[word] if word in d else word for word in words]
    reformed = " ".join(reformed)
    
    df['clean_tweet'] = df['clean_tweet'].apply(
        lambda x : ' '.join(d[word] if word in d else word for word in x.split()))
    
    #Finding words with # attached to it
    df['#'] = df['clean_tweet'].apply(
        lambda x : ' '.join([word for word in x.split() if word.startswith('#')]))
    frame = df['#']
    frame = pd.DataFrame(frame)
    frame = frame.rename({'#':'Count(#)'},axis = 'columns')
    frame[frame['Count(#)'] == ''] = 'No hashtags'
    data_frame = pd.concat([df,frame],axis = 1)
    data_frame.drop('#',axis = 1,inplace = True)
    
    #Column showing whether the corresponding tweet has a hash tagged word or not
    data_frame = data_frame.rename({'Count(#)':'Hash words'},axis = 'columns')

    #Removing stopwords
    data_frame['clean_tweet'] = data_frame['clean_tweet'].apply(
        lambda x : ' '.join([word for word in x.split() 
                             if word not in set(stopwords.words('english')) and len(word) > 3]))
    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    #data_frame['clean_tweet'] = data_frame['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    data_frame['clean_tweet'] = data_frame['clean_tweet'].apply(
        lambda x : ' '.join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in nltk.pos_tag(x.split())]))

    #Code to remove punctuation
    data_frame['clean_tweet'] = data_frame['clean_tweet'].apply(lambda x : re.sub('[%s]' % re.escape(string.punctuation),'',str(x)))
    
    #Stemming
    #ps = PorterStemmer()
    #adwait = data_frame
    #data_frame['clean_tweet'] = data_frame['clean_tweet'].apply(lambda x : ' '.join([ps.stem(word) for word in x.split()]))
    #reset index
    data_frame.reset_index(drop=True, inplace=True)
        
    return data_frame
```



<!-- EDA -->
## EDA

 <!-- word-clouds -->
 #### Word Clouds
 <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/word-clouds.png" width = "650" title = "word clouds for regions">
 
 <!-- data-overview -->
 #### Data Overview
 <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/percentage-of-speech-by-region.png" width = "300" title = "data overview">
 
 <!-- most-frequent-words -->
 #### Top Frequent Words
 <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/most-common-words.png" width = "950" title = "most frequent words">
 
 
<!-- sentiment-analysis-->
## Sentiment Analysis

  <!-- sentiment-result -->
  #### Sentiment Result
  <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/sentiment-scores.png" width = "400" title = "sentiment scores">
  
  <!-- polarity-result -->
  #### Polarity Result
  <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/polarity-score.png" width = "550" title = "polarity scores">
  
  <!-- subjectivity-result -->
  #### Subjectivity Result
  <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/subjectivity-score.png" width = "550" title = "subjectivity scores">
  
  <!-- emotion-result -->
  #### Emotion Result
  <img src = "https://github.com/merlinymy/Analytical-analysis-of-US-hate-crime-globally/blob/main/Plots/emotion-score.png" width = "750" title = "emotion scores">
  
<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
[1] (Croucher, 2020). Prejudice toward Asian Americans in the COVID-19 pandemic: the effects of social me- dia use in the United States. Frontiers in Communication, 5:39–39, 2020.

[2] (Gover, 2020). Anti-Asian hate crime during the COVID-19 pandemic: Exploring the reproduction of in- equality. American Journal of Criminal Justice, 45:647– 667, 2020.

[3] (Mu ̈ller and Schwarz, 2021). Karsten Mu ̈ller and Carlo Schwarz. Fanning the flames of hate: Social media and hate crime. Journal of the European Economic Association, 19:2131–2167, 2021.

[4] (Matthew and Williams, ). L Matthew and Williams. Hate in the machine: Anti-Black and anti-Muslim social media posts as predictors of offline racially and religiously aggravated crime. The British Journal of Criminology, 60:20–20.

[5] (Piatkowska and Messner). Piatkowska, S. J., Messner, S. F.,
 & Hövermann, A. (2020). Black out-group marriages
 and hate crime rates: A cross-sectional analysis of US
  
