# 🤖 FAQ Chatbot using NLP

An intelligent **FAQ Chatbot** built with Python, NLTK, spaCy, scikit-learn, and Gradio.
It preprocesses user queries, vectorizes them with **TF-IDF**, and finds the best
matching answer using **cosine similarity**.

---

## ✨ Features

- 50+ FAQ dataset covering AI, ML, Python, programming, technology, and college FAQs
- NLP preprocessing: lowercasing, tokenization, stopword removal, lemmatization
- TF-IDF vectorization + cosine similarity matching
- Confidence threshold with graceful fallback for unknown questions
- Modern dark-themed Gradio chat UI (ChatGPT-style bubbles)
- Typing animation, timestamps, confidence score, auto-scroll
- Search log storage (`logs/chat_logs.csv`)
- Built-in analytics (total queries, most-asked questions)
- Ready to run in **Google Colab** or locally

---

## 📁 Project structure

```
FAQ_Chatbot/
├── dataset/
│   └── faq.csv
├── notebooks/
│   └── chatbot.ipynb
├── utils/
│   ├── preprocessing.py
│   ├── vectorizer.py
│   ├── similarity.py
│   └── chatbot_engine.py
├── app/
│   └── gradio_app.py
├── logs/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation (local)

```bash
git clone <your-repo> FAQ_Chatbot
cd FAQ_Chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## ▶️ Run the chatbot

```bash
python app/gradio_app.py
```

Open the printed local URL (or the public `share` URL in Colab).

## 🧪 Run in Google Colab

1. Upload the `FAQ_Chatbot/` folder to Colab (or mount Drive).
2. Open `notebooks/chatbot.ipynb`.
3. Run all cells — the last cell launches the Gradio UI with `share=True`.

---

## 🧠 How it works

1. **Load** `dataset/faq.csv` (`question`, `answer` columns).
2. **Preprocess** each question — clean → tokenize → remove stopwords → lemmatize.
3. **Vectorize** preprocessed questions with TF-IDF (unigrams + bigrams).
4. On each user query, repeat the preprocessing and transform with the same vectorizer.
5. Compute **cosine similarity** between the query vector and all FAQ vectors.
6. Return the answer of the highest-scoring FAQ if `score >= 0.30`, otherwise return
   *"Sorry, I could not understand your question."*

---

## 🧰 Technologies used

| Layer        | Tools                                |
|--------------|--------------------------------------|
| Language     | Python 3.9+                          |
| NLP          | NLTK, spaCy                          |
| ML           | scikit-learn (TF-IDF, cosine sim)    |
| Data         | pandas, numpy, CSV                   |
| UI           | Gradio (dark theme, custom CSS)      |
| Platform     | Google Colab / local                 |

---

## 🖼️ Screenshots

_Add screenshots of the running Gradio UI here._

---

## 🧪 Testing examples

| Query                              | Expected behavior                              |
|------------------------------------|------------------------------------------------|
| `What is Machine Learning?`        | Returns the ML answer with high confidence     |
| `who created python`               | Returns the Guido van Rossum answer            |
| `asdkjasd random text`             | Returns the unknown-question fallback          |
| _(empty)_                          | Prompts the user to type a question            |

---

## 🔮 Future improvements

- Swap TF-IDF for **sentence-transformers** embeddings (semantic search)
- Multi-language support (translate query → answer → translate back)
- Speech-to-text & text-to-speech (SpeechRecognition + gTTS) — placeholders included
- GPT API fallback when confidence is low (OpenAI / Lovable AI Gateway)
- Admin dashboard for analytics
- Dockerfile + Hugging Face Space deployment

---

## 📄 License

MIT