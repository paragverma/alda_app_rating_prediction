import pandas as pd
from sklearn import preprocessing
import numpy as np
import datetime


  
import re
import time
import datetime
def processDf(r_df):
  df = r_df.copy(deep = True)
  # The best way to fill missing values might be using the median instead of mean.
  df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
  
  # Before filling null values we have to clean all non numerical values & unicode charachters 
  replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
  for i in replaces:
  	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))
  
  regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
  for j in regex:
  	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))
  
  df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)
  df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].median())
  i = df[df['Category'] == '1.9'].index
  df = df.drop(i)
  
  # Removing NaN values
  df = df[pd.notnull(df['Last Updated'])]
  df = df[pd.notnull(df['Content Rating'])]
  
  # App values encoding
  le = preprocessing.LabelEncoder()
  df['App'] = le.fit_transform(df['App'])
  # This encoder converts the values into numeric values
  
  # Category features encoding
  category_list = df['Category'].unique().tolist() 
  category_list = ['cat_' + word for word in category_list]
  df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)
  df = df.drop(['Category', category_list[0]], axis = 1)
  # Genres features encoding
  le = preprocessing.LabelEncoder()
  df['Genres'] = le.fit_transform(df['Genres'])
  
  # Encode Content Rating features
  le = preprocessing.LabelEncoder()
  df['Content Rating'] = le.fit_transform(df['Content Rating'])
  
  # Price cealning
  df['Price'] = df['Price'].apply(lambda x : x.strip('$'))
  
  # Installs cealning
  df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
  
  # Type encoding
  df['Type'] = pd.get_dummies(df['Type'])
  
  # Last Updated encoding
  df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))
  
  # Convert kbytes to Mbytes 
  k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()
  converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
  df.loc[k_indices,'Size'] = converter

  # Size cleaning
  df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
  s_indices = df[df['Size'] == 'Varies with device'].index
  #df.loc[s_indices, 'Size'] = 0
  #df['Size'] = df['Size'].astype(float)
  #df.loc[s_indices, 'Size'] = df['Size'].mean()
  #df[df['Size'] == 'Varies with device'] = 0
  #df['Size'] = df['Size'].astype(float)
  df = df.drop(s_indices, axis = 0)
  df.index = range(len(df))
  df['Size'] = df['Size'].astype(float)
  
  df['Reviews'] = df['Reviews'].astype(float)
  
  df['Installs'] = df['Installs'].astype(float)
  
  i = df[df['Current Ver'] == df['Current Ver'].max()].index
  df = df.drop(i)
  df.index = range(len(df))
  
  df = df.drop(['Android Ver', 'App'], axis = 1)
  
  return df


def getNumpyXy(dataset, y_column_name):
  y = dataset[y_column_name].values
  rating_index = list(dataset.columns).index(y_column_name)
  selector = [i for i in range(len(dataset.columns)) if i != rating_index]
  
  X = dataset.iloc[:, selector].values.astype(np.float64)
  
  return X, y