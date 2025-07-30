from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import os
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper to extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

# Clean and normalize text
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'\s+', ' ', text)    # normalize whitespace
    return text.strip().lower()

# Calculate similarity
def calculate_similarity(texts):
    texts = [clean_text(text) for text in texts]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return similarity_score

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis_results = None

    if request.method == 'POST':
        files = request.files.getlist('files')
        if len(files) != 2:
            analysis_results = {'error': 'Please upload exactly 2 HTML files.'}
        else:
            texts = []
            for file in files:
                html_content = file.read().decode('utf-8')
                text = extract_text_from_html(html_content)
                texts.append(text)

            similarity = calculate_similarity(texts)
            result = "⚠️ Possible Plagiarism Detected!" if similarity >= 50 else "✅ Documents appear to be original."

            analysis_results = {
                'similarity': f"{similarity:.2f}%",
                'result': result
            }

    return render_template('index.html', analysis_results=analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
