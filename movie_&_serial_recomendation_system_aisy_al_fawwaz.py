# -*- coding: utf-8 -*-
"""Movie & Serial Recomendation System_Aisy Al Fawwaz.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xhTov_VxwB_7gHN64epIRAgYDwUlsFqS

#Submission Machine Learning Terapan - Sistem Rekomendasi
**Nama : Aisy Al Fawwaz**

# 1. Menginstal library opendatasets
"""

!pip install opendatasets

"""# 2. Mengimpor library yang dibutuhkan"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np 
import opendatasets as od
import string
import re
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import cufflinks as cf
import plotly.io as pio
from wordcloud import WordCloud,STOPWORDS
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

"""#3. Mengunduh Dataset dari platfrom kaggle"""

od.download('https://www.kaggle.com/datasets/mazenramadan/imdb-most-popular-films-and-series')

"""# 4. Data Understanding

## 3.1 Menampilkan isi dari dataset dengan library pandas
"""

movies_meta_data = pd.read_csv('/content/imdb-most-popular-films-and-series/imdb.csv')

movies_meta_data

"""### 3.2 Menampilkan keterangan jpanjang data unique judul film"""

print(len(movies_meta_data.Name.unique()))

"""### 3.3 Menampilkan keterangan kolom dataset"""

movies_meta_data.info()

"""### 3.4 Menampilkan Daftar Genre pada dataset"""

print(movies_meta_data.Genre.unique())

"""### 3.5 menghitung panjang data pada variabel Genre"""

print(len(movies_meta_data.Genre.unique()))

"""### 3.6 Memuat deskripsi statistik pada setiap kolom dataframe """

# Memuat deskripsi setiap kolom dataframe
movies_meta_data.describe()

# Menghitung jumlah data kosong 
movies_meta_data.isnull().sum()

"""### 3.7 membuat variabel baru untuk dataset"""

# Memuat dataset ke dalam variable baru
movie = movies_meta_data.Name.unique()

# Mengurutkan data dan menghapus data yang sama
movie = np.sort(np.unique(movie))

print('Jumlah seluruh data movie berdasarkan movieId: ', len(movie))

movie_info = pd.concat([movies_meta_data])

movie_info

"""### 3.8 Visualisasi jumlah kata dengan frekuensi tertinggi pada kolom Genre"""

word_could_dict = Counter(movies_meta_data['Genre'].tolist())
frekuensi_kata = WordCloud(width = 3000, height = 1000).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(frekuensi_kata)
plt.title('frekuensi kata paling banyak muncul')
plt.axis("off")
plt.show()

"""# 4. Data Preparation

### 4.1 Memilih kolom berdasarkan data yang dibutuhkan dalam membuat sistem content based learning berdasarkan genre yaitu judul dan genre
"""

judul_film = movies_meta_data['Name'].tolist()
genre_film = movies_meta_data['Genre'].tolist()

print(len(judul_film))
print(len(genre_film))

"""### 4.2 Membuat data menjadi bentuk dataframe"""

data = pd.DataFrame({
    'judul': judul_film,
    'genre': genre_film
})

data.head(5)

data.info()

"""### 4.3 Memuat banyak data dari setiap unique value berdasarkan genre"""

value_genre = pd.DataFrame(data['genre'].value_counts().reset_index().values, columns = ['genre', 'count'])
print(len(value_genre))
pd.options.display.max_colwidth = 500
value_genre

# membuat data string tanda strip '-' pada variable data dihapus
data = data[data.genre != '-']

"""### Melihat kembali Jenis-Jenis Genre yang terdapat pada data"""

data.genre.unique()

"""### 4.3 Melakukan drop pada judul film yg double, dan berhasil menghapus beberapa judul"""

data = data.drop_duplicates('judul')
len(data)

"""**4.4 Melakukan indeks ulang pada data agar penomoran dilakukan berurutan**"""

data.reset_index()
data.head()

"""### 4.5 Memasukkan nilai data masing-masing kolom ke dalam variabel baru"""

judul = data['judul'].tolist()
genre = data['genre'].tolist()

print(len(judul))
print(len(genre))

# mengecek ulang data yg dimasukkan ke dalam variable baru
data = pd.DataFrame({
    'judul': judul,
    'genre': genre
})
data

"""## 4.6 Proses Data

### 4.6.1 Membangun model rekomendasi berdasarkan kesamaan genre
"""

# Inisialisasi CountVectorizer
tf = CountVectorizer()
 
# Melakukan perhitungan idf pada data genre
tf.fit(genre) 

# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names_out()

"""### 4.6.2 Melakukan Proses pelatihan pada model dan melihat ukuran matriks """

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(genre) 
 
# Melihat ukuran matrix tfidf
tfidf_matrix.shape

"""### 4.6.3 Mengubah vektor ke dalam bentuk matrix"""

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

"""### 4.6.4 Melihat Daftar jumlah film berdasarkan genre dan melihat korelasi nya yg diperlihatkan dalam bentuk matrix"""

pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names_out(),
    index=data.judul
).sample(22, axis=1).sample(10, axis=0)

"""# 5 Modeling

### 5.1 Melatih Model dengan cosine similarity
"""

cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

"""### 5.2 tahap ini menampilkan matriks kesamaan setiap judul dengan menampilkan judul film dalam 10 sampel kolom (axis = 1) dan 10 sampel baris (axis=0)."""

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['judul'], columns=genre)
print('Shape:', cosine_sim_df.shape)
 

cosine_sim_df.sample(10, axis=1).sample(10, axis=0)

"""# 6. Evaluasi Model

### 6.1 Pada tahap ini dilakukan indikasi dan diperlihatkan judul film berdasarkan urutan dari data
"""

indices = pd.Series(index = data['judul'], data = data.index).drop_duplicates()
indices.head()

"""### 6.2 Membuat fungsi untuk memanggil 5 rekomendasi film berdasarkan judul yang di input"""

def movie_recommendations(judul, cosine_sim = cosine_sim,items=data[['judul','genre']]):
    # Mengambil indeks dari judul film yang telah didefinisikan sebelumnnya
    idx = indices[judul]
    
    # Mengambil skor kemiripan dengan semua judul film 
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Mengurutkan film berdasarkan skor kemiripan
    sim_scores = sorted(sim_scores, key = lambda x : x[1], reverse = True)
    
    # Mengambil 5 skor kemiripan dari 0-5 karena urutan 0 memberikan indeks yang sama dengan judul film yang diinput
    sim_scores = sim_scores[0:5]
    
    # Mengambil judul film dari skor kemiripan
    movie_indices = [i[0] for i in sim_scores]
    
    # Mengembalikan 20 rekomendasi judul film dari kemiripan skor yang telah diurutkan dan menampilkan genre dari 20 rekomendasi film tersebut
    return pd.DataFrame(data['judul'][movie_indices]).merge(items)

# mengecek judul film di dalam data
data[data.judul.eq('King Kong')]

"""## 6.3 Mencoba menampilkan 5 rekomendasi film dari judul yang telah di input menggunakan fungsi movie_recomendations"""

recomendation = pd.DataFrame(movie_recommendations('King Kong'))
recomendation

# menghitung banyaknya data genre pada hasil rekomendasi yg dilakukan 
value = pd.DataFrame(recomendation['genre'].value_counts().reset_index().values, columns = ['genre', 'count'])
value.head()

"""### 6.4 Melakukan perhitungan dengan menggunakan metrik precision untuk melihat akurasi"""

TP = 5 #jumlah prediksi benar untuk genre yang mirip atau serupa
FP = 0 #jumlah prediksi salah untuk genre yang mirip atau serupa

Precision = TP/(TP+FP)
print("{0:.0%}".format(Precision))