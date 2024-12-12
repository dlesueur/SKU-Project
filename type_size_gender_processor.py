# create language processor model to assign a category to each SKU based on the description

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score
import shap
shap.initjs()
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import re

# read in data and define input and ouput variables
df = pd.read_csv('descriptionsAndTypes.csv')

types_to_exclude = ["Scrunchie", "Dog Supplement", "Alignment Sticks",
                    "Bandana", "Popcorn", "Makeup Tin", "Makeup Compact",
                    "Makeup Brush", "Pill Bottle", "Tincture", "Bar", "Coffee", "PIll Bottle"]

df = df[~df['type'].isin(types_to_exclude)]
df['type'] = df['type'].replace('Earbuds', 'Headphones')
df['type'] = df['type'].replace('tumbler', 'Tumbler')
df['type'] = df['type'].replace('Small Tool', 'Other')
df['type'] = df['type'].replace('Large Tool', 'Other')

X = df['description']
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# define stop words
with open('stopwords.txt') as f:
    sw = f.read().split()

# vectorize the text
vectorizer = TfidfVectorizer(min_df = 10, stop_words=sw)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


sm = SMOTE(random_state=307, k_neighbors=4)
X_train_aug, y_train_aug = sm.fit_resample(X_train_tfidf, y_train)

# fit random forest
rf_classifier = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=50, class_weight="balanced")
rf_classifier.fit(X_train_aug, y_train_aug)
print("Class order in rf_classifier:", rf_classifier.classes_)

# make predictions
y_train_pred = rf_classifier.predict(X_train_tfidf)
y_test_pred = rf_classifier.predict(X_test_tfidf)

# Train performance metrics
print("Training Metrics:")
print(classification_report(y_train, y_train_pred))
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Test performance metrics
print("\nTest Metrics:")
print(classification_report(y_test, y_test_pred))
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")


# Data to predict
new_df = pd.read_csv('SKUs.csv')
new_X_tfidf = vectorizer.transform(new_df['description'])
new_predictions = rf_classifier.predict(new_X_tfidf)
new_df['predicted_type'] = new_predictions






# add sex/line
def classify_line(description):
    description_lower = str(description).lower()
    if any(keyword in description_lower for keyword in ['mens', 'men', 'male', "men's"]):
        return 'Men'
    elif any(keyword in description_lower for keyword in ['women', 'ladies', "women's"]):
        return 'Women'
    else:
        return 'Uni'

new_df['line'] = new_df['description'].apply(classify_line)



new_df.to_csv('SKUs_details.csv', index=False)
