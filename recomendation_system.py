# -*- coding: utf-8 -*-
"""recomendation system.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/138eIAm0xFZyDAl3JSi_xDNtrtC000Iob
"""

import pandas as pd
import numpy as np
import seaborn as sns

book=pd.read_csv("book (1).csv", encoding='latin1')
book

book.head()

book1=book.drop("Unnamed: 0",axis=1)
book1

book2=book1.rename({'User.ID':'Userid', 'Book.Title':'Book_title', 'Book.Rating':'Book_Rating'},axis=1)

book2.info()

book2.shape

len(book2.Userid.unique())

len(book2.Book_Rating.unique())

len(book2.Book_title.unique())

book3 = book2.pivot_table(index='Userid',
                                 columns='Book_title',
                                 values='Book_Rating').reset_index(drop=True)

book3

book3.index = book2.Userid.unique()

book3

book3.fillna(0, inplace=True)
book3

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

user_sim = 1 - pairwise_distances(book3.values,metric='cosine')
user_sim

user_sim_df = pd.DataFrame(user_sim)
user_sim_df

user_sim_df.index = book2.Userid.unique()
user_sim_df.columns = book2.Userid.unique()

user_sim_df.iloc[0:7, 0:7]

user_sim_df.idxmax(axis=1)[0:5]

book2[(book2['Userid']==276729) | (book2['Userid']==162121)]

user_1=book2[book2['Userid']==276729] 
user_2=book2[book2['Userid']==162121]

user_1.Book_title

user_2.Book_title

