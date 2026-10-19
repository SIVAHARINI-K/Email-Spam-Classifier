import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
st.set_page_config(
    page_title="Smart Spam Detector",
    layout="centered"
)

st.markdown(
    """
    <h1 style='
    text-align: center;
    color: #FF4B4B;
    font-size: 50px;
    font-family: Montserrat;
    '>
    Smart Spam Detector
    </h1>
    """,
    unsafe_allow_html=True
)
df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

df = df.iloc[:, :2]

df.columns = ["label", "message"]

df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))

def transform_text(text):

    text = text.lower()

    text = text.split()

    y = []

    for i in text:

        word = ""

        for char in i:

            if char.isalnum():
                word += char

        if word != "":
            y.append(word)

    final_words = []

    for i in y:

        if i not in stop_words:
            final_words.append(ps.stem(i))

    return " ".join(final_words)

df["transformed_text"] = df["message"].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df["transformed_text"]).toarray()

y = df["label_num"]

model = MultinomialNB()

model.fit(X, y)

input_sms = st.text_area("Enter Message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:

        st.error("Spam Message")

    else:

        st.success("Ham Message")