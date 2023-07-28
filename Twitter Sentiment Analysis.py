#program for sentimental analysis of twitter account using python
#import the libraries

import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import re
from cleantext import clean
pl.style.use("fivethirtyeight")
import seaborn as sns




#twitter api credentials
consumerkey=""
consumersecret=""
accesstoken=""
accesstokensecret=""

#create the authentication object
authen=tweepy.OAuthHandler(consumerkey,consumersecret)

#set access
authen.set_access_token(accesstoken,accesstokensecret)

#create the api object
api=tweepy.API(authen)

#extract tweets from user
_id="103770785"
posts=api.user_timeline(screen_name="BillGates",count=100,tweet_mode="extended")
 
#print the 5 tweets from account
print("show five recent tweets:\n")
i=1
for tweet in posts[0:5]:
    print(str(i)+ ')'+tweet.full_text + '\n') 
    i=i+1

#create a dataframe
df=pd.DataFrame([tweet.full_text for tweet in posts] ,columns=["tweets"])

#show the first five row
df.head

def clean(text):
    text=re.sub(r'@[a-zA-Z0-9]+','',text)
    text=re.sub(r'#','',text)
    text=re.sub(r'RT[\s]+','',text)
    text=re.sub(r'https?:\/\/\S+','',text)#remove hyperlink
    return text
df["tweets"]=df['tweets'].apply(lambda x:clean(x)) #recursive function
df.head(1)


print(df)

#show the cleaned text


#subjectivity
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#polarity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity

#create two columns
df['subjectivity']=df['tweets'].apply(getsubjectivity)
df['polarity']=df['tweets'].apply(getpolarity)

#show df
df
#plot the wordcloud
allwrds=' '.join([tweet for tweet in df['tweets']])
wrdcld=WordCloud(width=500,height=300).generate(allwrds)

pl.imshow(wrdcld,interpolation='bilinear')
pl.axis('off')
# pl.show()

#create func to know neg,pos and neutral analysis
def getanalyse(score):
    if score <0:
        return 'negative'
    elif score==0:
        return 'neutral' 
    else:
        return 'positive'

df['analysis']=df['polarity'].apply(getanalyse)

#show the df
df

#print all the positive tweets
j=1
sorteddf=df.sort_values(by=['polarity'])
for i in range(0,sorteddf.shape[0]):
    if(sorteddf['analysis'][i]=='positive'):
        print(str(j) +')'+sorteddf['tweets'][i])
        print()
        j=j+1

#print all neg twets 
j=1
sorteddf=df.sort_values(by=['polarity'],ascending=False)
for i in range(0,sorteddf.shape[0]):
    if(sorteddf['analysis'][i]=='negative'):
        print(str(j)+')'+sorteddf['tweets'][i])
        print()
        j=-j+1

#plot the sub and pol
pl.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
    pl.scatter(df['polarity'][i],df['subjectivity'][i],color='blue')

pl.title('SENTIMENT ANALYSIS')
pl.xlabel('polarity')
pl.ylabel('subjectivity')
pl.show()

#pecentage of positive tweets
ptweets=df[df.analysis=='positive']
ptweets=ptweets['tweets']

print(ptweets.shape[0]/df.shape[0]*100,1)

#pecentage of negative tweets
ntweets=df[df.analysis=='negative']
ntweets=ntweets['tweets']

print(ntweets.shape[0]/df.shape[0]*100,1)

#show the value count 
df['analysis'].value_counts()
# print(df)
#plot and visualise the count
pl.title("SENTIMENT ANALYSIS")
pl.xlabel('sentiment')
pl.ylabel('counts')
df['analysis'].value_counts().plot(kind="bar")
pl.show()

