import sys
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pickle



def load_data(database_filepath):
    
    engine = create_engine('sqlite:///'+ str (database_filepath))
    df = pd.read_sql ('SELECT * FROM Messages', engine)
    X = df ['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X,y,category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]

    return lemmed
    


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'vect__max_features': [5000, 10000, 20000],  # Number of features for CountVectorizer
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],  # N-gram range for CountVectorizer
        'tfidf__use_idf': [True, False],  # Whether to use IDF in TfidfTransformer
        'tfidf__norm': ['l1', 'l2'],  # Normalization for TfidfTransformer
        'clf__n_estimators': [100, 200, 300],  # Number of trees in RandomForestClassifier
        'clf__max_depth': [None, 5, 10],  # Maximum depth of trees in RandomForestClassifier
        'clf__min_samples_split': [2, 5, 10],  # Minimum samples for split in RandomForestClassifier
        'clf__min_samples_leaf': [1, 5, 10]  # Minimum samples for leaf in RandomForestClassifier
    }

    model = GridSearchCV(pipeline, parameters, cv=5, scoring='f1_macro')
    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()