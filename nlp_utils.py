import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')   # ✅ ADD THIS LINE
nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    tokens = [word for word in tokens if word.isalnum()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)