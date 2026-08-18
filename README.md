# Hotel Review Sentiment Analysis

## 📌 Project Overview

This project performs **sentiment analysis on hotel reviews** using Natural Language Processing (NLP) and Machine Learning.

The objective is to classify hotel reviews into sentiment categories based on the `Is_Response` field. The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction and **Logistic Regression** for classification.

---

## 📂 Dataset

The project uses a dataset named:

`hotel-reviews.csv`

The notebook loads the dataset using Pandas and performs initial data exploration, including:

* Dataset information
* Descriptive statistics
* Missing-value checking
* Column inspection
* Sentiment distribution analysis

Some unnecessary columns are removed during preprocessing:

* `User_ID`
* `Browser_Used`
* `Device_Used`

The main text column used for sentiment analysis is:

* `Description`

The target column is:

* `Is_Response`

---

## 🔄 Project Workflow

The analysis follows these major steps:

### 1. Import Libraries

The project uses libraries such as:

* Pandas
* NumPy
* Regular Expressions (`re`)
* Matplotlib
* Seaborn
* Scikit-learn

### 2. Load the Dataset

The hotel review dataset is loaded into a Pandas DataFrame.

```python
data = pd.read_csv("hotel-reviews.csv")
```

### 3. Exploratory Data Analysis

The notebook examines:

* Dataset structure
* Statistical information
* Missing values
* Available columns
* Distribution of sentiment classes

A count plot is also used to visualize the distribution of `Is_Response`.

### 4. Data Cleaning

The review text is cleaned using regular expressions.

The first cleaning function:

* Converts text to lowercase
* Removes text inside brackets
* Removes punctuation
* Removes words containing digits

A second cleaning function is also applied to remove selected unwanted characters and newline characters.

The cleaned review text is stored in:

`clean_description`

and

`clean_description_new`

### 5. Train-Test Split

The cleaned review descriptions are separated into input (`X`) and target (`y`).

The dataset is split into:

* **70% training data**
* **30% testing data**

A `random_state` of `42` is used to make the split reproducible.

### 6. TF-IDF Feature Extraction

The project uses `TfidfVectorizer` to convert textual hotel reviews into numerical features.

TF-IDF gives greater importance to words that are useful for distinguishing between different review categories.

### 7. Logistic Regression Model

A Logistic Regression classifier is used for sentiment classification.

The TF-IDF vectorizer and Logistic Regression classifier are combined into a Scikit-learn `Pipeline`.

```python
model = Pipeline([
    ('vectorzer', tvec),
    ('classifier', clf)
])
```

The model is then trained using the training dataset.

### 8. Model Prediction

After training, the model predicts sentiment labels for the test dataset.

```python
pred = model.predict(x_test)
```

### 9. Model Evaluation

The notebook evaluates the classifier using:

* Confusion Matrix
* Classification Report

The classification report provides metrics such as:

* Precision
* Recall
* F1-score
* Support

### 10. Example Prediction

The trained model is also tested with a new example sentence:

```python
example = ['Point out reasons the visit was not favorable']
pre = model.predict(example)
```

This demonstrates how the trained model can be used to predict the sentiment category of previously unseen text.

---

## 🛠️ Technologies Used

| Technology          | Purpose                              |
| ------------------- | ------------------------------------ |
| Python              | Programming language                 |
| Pandas              | Data loading and manipulation        |
| NumPy               | Numerical operations                 |
| Matplotlib          | Data visualization                   |
| Seaborn             | Sentiment distribution visualization |
| Scikit-learn        | Machine learning and evaluation      |
| TF-IDF              | Text feature extraction              |
| Logistic Regression | Sentiment classification             |

---

## 📊 Machine Learning Pipeline

```text
Hotel Reviews Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Remove Unnecessary Columns
        ↓
Text Cleaning
        ↓
Train-Test Split
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Sentiment Prediction
        ↓
Model Evaluation
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Make sure the dataset is available

Place `hotel-reviews.csv` in the same directory as the notebook.

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
hotelReview_Sentimental_analysis.ipynb
```

and run the cells sequentially.

---

## 📁 Project Structure

```text
.
├── hotelReview_Sentimental_analysis.ipynb
├── hotel-reviews.csv
└── README.md
```

---

## 🎯 Objective

The primary goal of this project is to demonstrate how **NLP and supervised machine learning** can be applied to hotel reviews to automatically identify their sentiment.

The project provides an end-to-end workflow covering:

**Data preprocessing → Text cleaning → Feature extraction → Model training → Prediction → Evaluation**

---

## 🔮 Possible Improvements

The current notebook can be extended by:

* Comparing Logistic Regression with Naive Bayes, SVM, or other classifiers
* Adding stop-word removal
* Applying stemming or lemmatization
* Performing hyperparameter tuning
* Comparing different TF-IDF configurations
* Adding accuracy and confusion-matrix visualizations
* Handling class imbalance if present
* Deploying the trained model as a web application or API

---

## 👤 Author

**Hotel Review Sentiment Analysis Project**

Built using Python, NLP, and Machine Learning.
