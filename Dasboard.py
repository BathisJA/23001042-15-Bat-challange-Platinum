import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from PIL import Image

# Load datasets
df_train = pd.read_csv('Data Training/train_preprocess.tsv', sep='\t', names=['text', 'label'])
df_slang = pd.read_csv('DataKlasifikasi/new_kamusalay.csv', encoding='latin-1', names=['alay', 'kbbi'])
df_test = pd.read_csv('DataKlasifikasi/data.csv', encoding='latin-1', usecols=['Tweet'])
df_test.rename(columns={'Tweet': 'text'}, inplace=True)
df_abusive = pd.read_csv('DataKlasifikasi/abusive.csv', encoding='latin-1')

# Function for data cleaning
def clean_character(text):
    string = text.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    string = re.sub(r'\brt\b', ' ', string)
    string = re.sub(r'\buser\b', ' ', string)
    string = re.sub(r'\burl\b', ' ', string)
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', string)
    string = string.encode('unicode_escape').decode('utf-8')
    string = re.sub(r'x[a-zA-Z0-9]{2}', '', string)
    string = string.encode('utf-8').decode('unicode_escape')
    string = re.sub(':', ' ', string)
    string = re.sub(';', ' ', string)
    string = re.sub(r'\+n', ' ', string)
    string = re.sub('\n', " ", string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub('  +', ' ', string)
    return string

# Define alay_dict_map
alay_dict_map = dict(zip(df_slang['alay'], df_slang['kbbi']))
# Function for alay cleansing
def alay_cleanse(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

# Function for censoring abusive words
def censor_abusive_words(text):
    cleaned_text = alay_cleanse(text)
    abusive_words = df_abusive['ABUSIVE']
    for word in abusive_words:
        if word in cleaned_text:
            cleaned_text = cleaned_text.replace(word, word[0] + '*' * (len(word) - 2) + word[-1])
    return cleaned_text

# Apply cleaning functions to train dataset
df_train['cleaned_text'] = df_train['text'].apply(clean_character)
df_train['cleaned_text'] = df_train['cleaned_text'].apply(alay_cleanse)
df_train['cleaned_text'] = df_train['cleaned_text'].apply(censor_abusive_words)

# Apply cleaning functions to test dataset
df_test['cleaned_text'] = df_test['text'].apply(clean_character)

df_train['total char'] = df_train['text'].apply(lambda x: len(clean_character(x)))
df_train['total word'] = df_train['text'].apply(lambda x: len(clean_character(x).split()))


st.title('DOKUMENTASI PROJECT BOOTCAMP BINAR')

st.title('Exploratory Data Analysis')
st.markdown("""
    Ini adalah dokumentasi dari proyek Bootcamp Data Science dari Binar Academy. 
    
    **Tujuan Proyek:**
            
    Melakukan analisis sentimen terhadap data Twitter menggunakan metode Neural Network dan Long Short Term Memory melalui API.
            
    Dibawah ini adalah hasil dari analisis data yang telah dilakukan beserta contoh API yang sudah dibuat.
""")

st.sidebar.header('Detail Dataset')
st.sidebar.markdown('Ini adalah detail mengenai dataset yang digunakan unutk melakukan training pada Machine Learning')

st.sidebar.subheader('Null Values')
st.sidebar.write(df_train.isnull().sum().rename_axis('Label').reset_index(name='Jumlah'), index=False)

st.sidebar.subheader('Duplicate Values')
st.sidebar.write("Data duplikat:", df_train.duplicated().sum())

st.sidebar.subheader("Rata-rata Jumlah Kata")
st.sidebar.write(df_train[['total char', 'total word']].mean())

st.sidebar.subheader("Distribusi Label di Dataset")
st.sidebar.write(df_train['label'].value_counts())


st.subheader('Data Yang Sudah Dibersihkan')
st.markdown('Berikut ini adalah contoh data yang sudah dilakukan proses cleansing, yaitu mengganti kata slang dengan kata baku dan melakukan sensor terhadap kata kasar, beserta label sentimen dari data tersebut. Data ini nantinya digunakan unutk melakukan training Machine Learning.')
st.write(df_train[['cleaned_text', 'label']].head(21))

st.subheader('Proporsi Label')
image_1 = Image.open('Templates/piechart.png')
st.image(image_1, use_column_width=True)
st.write("Pie chart ini menggambarkan distribusi label data train, dimana label tersebut terbagi menjadi positif, negatif, dan netral.")
st.markdown("""
- **Positive:** 58.4%
- **Negative:** 10.4%
- **Neutral:** 31.2%
""")


st.subheader('Persebaran Tweet')
image_2 = Image.open('Templates/barchart.png')
st.image(image_2, use_column_width=True)
st.write("Secara keseluruhan, grafik menunjukkan bahwa tweet dalam dataset tersebut relatif pendek, dengan kebanyakan tweet memiliki kurang dari 400 karakter dan kurang dari 60 kata.")


st.subheader('Word Cloud Kata Kasar')
image_3 = Image.open('Templates/wordcloud_abusive.png')
st.image(image_3, use_column_width=True)
st.write("Ini menunjukan kata-kata kasar yang sering digunakan dalam dataset.")


st.subheader('Word Cloud Kata Kasar')
image_4 = Image.open('Templates/wordcloude_alay.png')
st.image(image_4, use_column_width=True)
st.write("Ini menunjukan kata-kata alay yang sering digunakan dalam dataset.")


st.title('Model Machine Learning')
st.markdown('Dibawah ini adalah visualisasi hasil evaluasi dari machine learning yang sudah dibuat.')

st.subheader('Model LSTM')
image_5 = Image.open('Templates/lstm.png')
st.image(image_5, use_column_width=True)
st.write("Layer yang kami gunakan dalam model:")
st.markdown("""
- KFold Cross Validation
- Sequential model 
- Embedding Layer
- LSTM Layer
- Dense Layer
- Optimizer dan Loss Function:
- EarlyStopping
""")


st.subheader('Model NN')
image_6 = Image.open('Templates/nn.png')
st.image(image_6, use_column_width=True)
st.write("Layer yang kami gunakan dalam model:")
st.markdown("""
- KFold Cross Validation
- Sequential model 
- Embedding Layer
- SimpleRNN Layer
- Dense Layer
- Optimizer dan Loss Function:
- EarlyStopping
""")

st.title('Tampilan API')
st.markdown("""
    Ini adalah contoh API yang dibuat menggunakan Flask dan Swager. 
    Tampilan API diubah menggunakan HTML, CSS, dan JavaScript agar lebih menarik dan mudah digunakan.
""")

st.subheader('Tampilan Asli')
image_7 = Image.open('Templates/swagger.png')
st.image(image_7, use_column_width=True)

st.subheader('Landing Page')
image_7 = Image.open('Templates/Landing Page.png')
st.image(image_7, use_column_width=True)

st.subheader('Input Page')
image_8 = Image.open('Templates/Input Page.png')
st.image(image_8, use_column_width=True)

st.subheader('Contoh Output')
image_9 = Image.open('Templates/api.png')
st.image(image_9, use_column_width=True)
