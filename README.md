# 📧 Spam Detector App
Welcome to the Spam Detector — a machine learning-powered web app that classifies email messages as Spam 🚫 or Not Spam, using natural language processing and the XGBoost algorithm.

## Live Demo

Check out the live app here:  
👉 [Spam Email Detection App](https://spam-email-detection-js7dfonwtilwxchxmsap3g.streamlit.app/)

## Acknowledgement
I want to use this privilege to appreciate Muhammad Abdullah, the author of "NLP Email Spam Detection: A Beginner's Guide." His work has been instrumental in helping me understand the complexities of NLP and its practical applications. The open documentation he has provided was so effective that I was able to replicate his work and successfully apply it to my own dataset. I am truly grateful for his contributions.

## Features
-  Real-time spam classification
- NLP preprocessing with stopword removal, punctuation stripping, and tokenization
-  Trained on a labeled email dataset using multiple classifiers
- Final model: XGBoost with 97.6% accuracy and 100% precision
- Built with Streamlit for an interactive web interface

  ## Methodology
Natural Language Processing (NLP) tasks using Machine Learning techniques

##### steps:
- Data Preprocessing: We'll discuss how to prepare and clean the data for NLP tasks. This includes tasks such as removing special characters, handling capitalization, tokenization, and dealing with stopwords.
- Data Vectorization: Next, we'll explore various methods for converting text data into numerical representations suitable for machine learning models. This includes techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
- Model Training: We'll dive into training different machine learning models for NLP tasks. This involves selecting appropriate models like Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, Random Forests, and more. We'll evaluate their performance using metrics like accuracy, precision.
a. Training and Evaluation: We'll train each model and evaluate its performance using various evaluation metrics to understand how well it generalizes to unseen data.
b. Model Selection: Finally, we'll identify the model that demonstrates the best performance on our dataset and discuss strategies for model selection.

###  Text Preprocessing

| Library | Function |
|--------|----------|
| `string` | Provides tools to handle and manipulate strings, including punctuation removal. |
| `re` | Regular expressions for pattern matching and text cleaning. |
| `nltk.corpus.stopwords` | Contains lists of stopwords (common words like "the", "is") to remove from text. |
| `nltk.stem.porter.PorterStemmer` | Reduces words to their root form (e.g., "running" → "run"). |

---

### Data Handling & Visualization

| Library | Function |
|--------|----------|
| `numpy` | Supports numerical operations and array handling. |
| `pandas` | Used for data manipulation and analysis with DataFrames. |
| `matplotlib.pyplot` | Basic plotting library for visualizing data. |
| `seaborn` | Built on matplotlib; provides more attractive and informative statistical graphics. |

---

###  Feature Extraction

| Library | Function |
|--------|----------|
| `CountVectorizer` | Converts text to a matrix of token counts (Bag of Words model). |
| `TfidfVectorizer` | Converts text to a matrix of TF-IDF features (term importance). |

---
###  Machine Learning Models

| Library | Function |
|--------|----------|
| `LogisticRegression` | Linear classifier for binary/multiclass classification. |
| `SVC` | Support Vector Classifier for separating data with hyperplanes. |
| `GaussianNB`, `MultinomialNB`, `BernoulliNB` | Naive Bayes classifiers for different data distributions. |
| `DecisionTreeClassifier` | Tree-based model that splits data based on feature values. |
| `KNeighborsClassifier` | Classifies based on the majority label of nearest neighbors. |
| `RandomForestClassifier` | Ensemble of decision trees for better accuracy and robustness. |
| `AdaBoostClassifier` | Boosts weak learners sequentially to improve performance. |
| `BaggingClassifier` | Trains multiple models on random subsets of data to reduce variance. |
| `ExtraTreesClassifier` | Similar to Random Forest but uses more randomness in tree splits. |
| `GradientBoostingClassifier` | Builds models sequentially to correct previous errors. |
| `XGBClassifier` | Optimized gradient boosting library for high performance. |

---

###  Model Evaluation & Splitting

| Library | Function |
|--------|----------|
| `train_test_split` | Splits data into training and testing sets. |
| `sklearn.metrics` | Provides tools to evaluate model performance (accuracy, precision, recall, etc.). |


# How It Works
- Preprocessing: Cleans and normalizes the input text
- Vectorization: Converts text into numerical features using CountVectorizer
- Prediction: Uses a trained XGBoost model to classify the message
- Output: Displays whether the message is spam or not

## Installation
```
git clone https://github.com/your-username/spam-detector
cd spam-detector
pip install -r requirements.txt
```

#### Run the App
```
streamlit run app.py
```

## Files Included
- app.py — Streamlit app
- spam_classifier.pkl — Trained XGBoost model
- vectorizer.pkl — Fitted CountVectorizer
- requirements.txt — Python dependencies
- README.md — Project overview


## Technologies Used
- Python
- Streamlit
- NLTK
- Scikit-learn
- XGBoost

## 👤 Author
Peace Adamu, Nigeria Built with machine learning
