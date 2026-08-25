# 🏨 Hotel Review Sentiment Analysis

A Natural Language Processing (NLP) and Machine Learning project that classifies hotel reviews as **Happy** or **Not Happy** using **TF-IDF and Logistic Regression**.

## 📌 Project Overview

Hotel reviews contain valuable information about customer satisfaction. This project automatically analyzes hotel review text and predicts whether the customer is **Happy** or **Not Happy**.

The complete pipeline is:

Dataset → Text Cleaning → Preprocessing → TF-IDF → Train/Test Split → Logistic Regression → Evaluation → Prediction

## 📊 Dataset

The project uses a hotel review dataset containing **38,932 reviews**.

### Important Columns

| Column | Description |
|---|---|
| `User_ID` | Unique user identifier |
| `Description` | Hotel review text |
| `Browser_Used` | Browser used by the reviewer |
| `Device_Used` | Device used by the reviewer |
| `Is_Response` | Target sentiment: Happy / Not Happy |

The model uses:

- `Description` → Input
- `Is_Response` → Target

### Class Distribution

| Class | Reviews |
|---|---:|
| Happy | 26,521 |
| Not Happy | 12,411 |

## 🔄 Methodology

### 1. Data Loading

The hotel review CSV file is loaded using Pandas.

### 2. Data Cleaning

Missing reviews and sentiment labels are removed.

### 3. Text Preprocessing

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove URLs
- Remove numbers
- Remove punctuation
- Tokenize text
- Remove English stopwords
- Apply lemmatization using NLTK WordNet

### 4. TF-IDF Feature Extraction

TF-IDF is used to convert text into numerical features.

Configuration:

```python
max_features = 10000
ngram_range = (1, 2)
