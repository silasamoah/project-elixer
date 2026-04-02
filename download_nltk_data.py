# download_nltk_data.py (run once with network)
import nltk

print("📥 Downloading NLTK data...")
nltk.download("punkt")
nltk.download("punkt_tab")  # Python 3.12+ needs this too
nltk.download("stopwords")  # if you use it
print("✅ NLTK data downloaded!")