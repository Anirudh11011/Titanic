import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("🚢 Titanic Survival Prediction - Model Evaluation")

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def preprocess_data(train):
    full_data = [train]

    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)

        dataset['Title'] = dataset['Name'].apply(lambda name: re.search(' ([A-Za-z]+)\.', name).group(1) if re.search(' ([A-Za-z]+)\.', name) else "")
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
        title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0).astype(int)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'Has_Cabin']
    X = train[features]
    y = train['Survived']
    return train, X, y

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', max_depth=5)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 3. AdaBoost
    ab = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50, learning_rate=0.01, random_state=42
    )
    ab.fit(X_train, y_train)
    y_pred_ab = ab.predict(X_test)

    models = ['Decision Tree', 'Random Forest', 'AdaBoost']
    predictions = [y_pred_dt, y_pred_rf, y_pred_ab]

    metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    for name, preds in zip(models, predictions):
        metrics['Model'].append(name)
        metrics['Accuracy'].append(accuracy_score(y_test, preds))
        metrics['Precision'].append(precision_score(y_test, preds))
        metrics['Recall'].append(recall_score(y_test, preds))
        metrics['F1 Score'].append(f1_score(y_test, preds))

    return pd.DataFrame(metrics)

if st.button("🔍 Run Models"):
    with st.spinner("Training and evaluating models..."):
        train, test = load_data()
        _, X, y = preprocess_data(train)
        results = evaluate_models(X, y)
    st.success("Done!")
    st.subheader("📊 Evaluation Results")
    st.dataframe(results)
