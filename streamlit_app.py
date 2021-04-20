import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS

import nltk
nltk.download('vader_lexicon') 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Disable the warnings.
st.set_option('deprecation.showPyplotGlobalUse', False)
                      

# Some utility functions.
def build_wordcloud(df, title):
  wordcloud = WordCloud(
      background_color='gray', 
      stopwords=set(STOPWORDS), 
      max_words=50, 
      max_font_size=40, 
      random_state=666
  ).generate(str(df))

  fig = plt.figure(1, figsize=(14,14))
  plt.axis('off')
  fig.suptitle(title, fontsize=16)
  fig.subplots_adjust(top=2.3)

  plt.imshow(wordcloud)
  st.pyplot()
  
def categorize(senti):
  if senti < 0:
    return 'Negative'
  elif senti == 0:
    return 'Neutral'
  else:
    return 'Positive'
  
# Global Variables
ORDER_OF_AGREE = ['Strongly agree','Agree','Slightly agree','Slightly disagree','Disagree','Strongly disagree']
ORDER_OF_AGREE2 = ['Strongly Agree','Agree','Slightly Agree','Moderately Agree','Moderately Disagree','Slightly Disagree','Strongly Disagree']
 
# Set the title and basic layout of the application.
st.title("PRU Feedback Survey Dashboard")
st.markdown('_Exploratory Analysis_ | Powered by **PRU Feedback Team**')

st.sidebar.header('Dashboard for PRU Feedback Survey')

with st.sidebar.beta_expander("Instructions"):
  st.markdown('''
    * The whole process is grouped into 4 parts: **time series analysis**, **demographic analysis**, **multiple-choice question analysis** and **textual analysis**. 
    * Please follow the logic flow and do not miss any button to ensure the normal operation. 
    * Also the code may be reused for future surveys upon minor modifications.
                      ''')

# Read in the data
data = pd.read_csv('sample.csv')

# Number of responses along the timeline

# with st.beta_expander('Time Series Analysis'):
#   data.Timestamp = data.Timestamp.apply(lambda x:pd.to_datetime(x[:10]))
#   dates = data.Timestamp.value_counts().sort_index()

#   timeindex = pd.date_range(start=dates.index[0],end=dates.index[-1])
#   all_dates = pd.DataFrame(columns=['Number_of_replies'],index=timeindex)

#   for i in timeindex:
#     if i in dates.index:
#       all_dates.loc[i,'Number_of_replies'] = dates[i]
#     else:
#       all_dates.loc[i,'Number_of_replies'] = 0

#   st.subheader("Response Timeline")
  
#   st.dataframe(all_dates)
#   st.line_chart(all_dates)
  

with st.beta_expander("Demography Analysis"):
  
  # Data Preparation
  data['count'] = 1
  
  data.rename({'Which Faculty are you from? (Indicate your home faculty if you are in a double degree programme.)':'Faculty',
              'Which Year of Study are you currently in?': 'Year of Study'}, axis='columns',inplace=True)
  df_faculty = data[['Faculty','count']].groupby(['Faculty']).count().reset_index()
  df_year = data[['Year of Study','count']].groupby(['Year of Study']).count().reset_index()
  
  st.subheader('Distribution of Respondent Home Faculty')
  fig = px.pie(
      df_faculty, 
      names="Faculty", 
      values="count", 
      color="Faculty",
      width=600, height=600
  )
  st.plotly_chart(fig)
  
  st.subheader('Distribution of Respondent Year of Study')
  fig = px.bar(
      df_year, 
      x="count", 
      y="Year of Study", 
      color = "Year of Study",
      orientation='h', 
      width=600, height=600
  )
  
  st.plotly_chart(fig)  
  
with st.beta_expander('Textual Analysis'):
  
  df_zoning = data['I feel that the zoning restrictions were ________.'].rename_axis('comments')

  st.header('I feel that the zoning restrictions were ____.')
  if st.checkbox('Show n random comments',True,key='1'):
    number = st.slider('Number of comments to take a look at:', 1, df_zoning.dropna().shape[0],3,key='1')
    sample = df_zoning.dropna().sample(number)
    st.table(sample)
  
  st.subheader('Wordcloud')
 
   
  clouds = st.slider('Select n comments to visualize their collective word clouds',1,df_zoning.dropna().shape[0],key='select')
  if st.button('Click to show the cloud',key='cloud'):
      build_wordcloud(df_zoning.dropna().sample(clouds), f'Word Cloud for {clouds} sampled comments')
      
    
  
  st.subheader('Sentiment Analysis')
  st.markdown('The sentiment is evaluated as a polarity score ranging from -1 to 1. The lower the comment is scored, the less positive it tends to be.')
  
  sid = SentimentIntensityAnalyzer()
  scores = pd.Series([sid.polarity_scores(str(text))['compound'] for text in df_zoning.values],name='score')
  df_zoning_score = pd.concat([df_zoning, scores],axis=1)
  df_zoning_score.columns = ['comments','scores']
  if st.checkbox('Show some examples',False,key='score'):
    st.table(df_zoning_score.sample(5))
    
  df_zoning_score['category'] = df_zoning_score.scores.apply(lambda x:categorize(x))
  
  if st.checkbox('Show the boxplot of Polarity Score',True,key='boxplot'):
    fig = px.box(df_zoning_score, x="category", y="scores", points="all")
    st.plotly_chart(fig)
    
  st.subheader('Categorize the sentiment')
  st.markdown('To make easier comparison, the sentiment is categoried into \'positive\', \'neutral\' and \'negative\' based on their polarity score. Now it\'s time to visualize our findings.')
  
  combined_with = st.multiselect('Combine the result with', ('Faculty','Year of Study'),key='combo')
  
  if 'Faculty' in combined_with:
    faculties = data.Faculty.unique()
    df_fac_sen = pd.concat([df_zoning_score.category, data.Faculty],axis=1)
    df_fac_sen['count'] = 1
    fac_pos = df_fac_sen.groupby(['category','Faculty']).count()
    counts = pd.DataFrame(columns = ['Negative','Neutral','Positive'], index=faculties)
    
    for i in counts.columns:
      for j in counts.index:
        if (i,j) not in fac_pos.index:
          counts.loc[j,i] = 0
        else:
          counts.loc[j,i] = fac_pos.loc[i,j]['count']
   
   
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Negative', x=faculties, y=counts['Negative'],marker_color='red'))
    fig.add_trace(go.Bar(name='Neutral', x=faculties, y=counts['Neutral'],marker_color='grey'))
    fig.add_trace(go.Bar(name='Positive', x=faculties, y=counts['Positive'],marker_color='green'))
    
    fig.update_layout(
      title='Sentiment Analysis Across Faculties',
      xaxis_tickfont_size=15,
      yaxis=dict(
          title='Counts',
          titlefont_size=16,
          tickfont_size=14,
      ),
      barmode='group',
      bargap=0.15, # gap between bars of adjacent location coordinates.
      bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    # Change the bar mode
    st.plotly_chart(fig)    
  
  if 'Year of Study' in combined_with:
    years = data['Year of Study'].unique()
    df_year_sen = pd.concat([df_zoning_score.category, data['Year of Study']],axis=1)
    df_year_sen['count'] = 1
    year_pos = df_year_sen.groupby(['category','Year of Study']).count()
    counts = pd.DataFrame(columns = ['Negative','Neutral','Positive'], index=years)
    
    for i in counts.columns:
      for j in counts.index:
        if (i,j) not in year_pos.index:
          counts.loc[j,i] = 0
        else:
          counts.loc[j,i] = year_pos.loc[i,j]['count']    
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Negative', x=years, y=counts['Negative'],marker_color='red'))
    fig.add_trace(go.Bar(name='Neutral', x=years, y=counts['Neutral'],marker_color='grey'))
    fig.add_trace(go.Bar(name='Positive', x=years, y=counts['Positive'],marker_color='green'))
    
    fig.update_layout(
      title='Sentiment Analysis Grouped by Years of Study',
      xaxis_tickfont_size=15,
      yaxis=dict(
          title='Counts',
          titlefont_size=16,
          tickfont_size=14,
      ),
      barmode='group',
      bargap=0.15, # gap between bars of adjacent location coordinates.
      bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    # Change the bar mode
    st.plotly_chart(fig)    
    
    
  if st.checkbox('Overall distribution',False,key='overall'):
    df_zoning_score['count'] = 1
    df_sen = df_zoning_score.groupby(['category']).count().reset_index()
    
    fig = px.pie(
      df_sen, 
      names="category", 
      values="count", 
      color_discrete_sequence = ['red','grey','green'],
      title='Sentiment Analysis in General'
    )
    
    st.plotly_chart(fig)    

with st.beta_expander('Multiple Choice Question Analysis (Example)'):
  
  data.rename({'During my time on campus,the COVID-19 measures put in place by NUS were sufficient in ensuring the safety of our health.':'Measures easy to follow'},inplace=True,axis=1)
  df = data[['Measures easy to follow','count']].groupby(['Measures easy to follow']).count().reset_index()
  df['sort'] = df['Measures easy to follow'].apply(lambda x:ORDER_OF_AGREE2.index(x))
  df.sort_values(by=['sort'])
  fig = px.bar(
    df, 
    x="Measures easy to follow", 
    y="count", 
    color = "Measures easy to follow",
    title='The COVID-19 measures put in place by NUS were easy to follow.'
  )
  st.plotly_chart(fig)  
  
  st.text('To be continued...')
  