import sys
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt') 
import pickle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer




def load_data(database_filepath):
    '''
    Loads data from an SQLite database and splits it into features and target variables.
    
    Input:
    - database_filepath: SQLite database that contains messages and their categories.

    Output:
    - X: The feature data, specifically the 'message' column from the database.
    - y: The target data, containing all the category columns starting from the 5th column onward.
    - category_names: The names of the categories corresponding to the target variables.
    '''
    engine = create_engine('sqlite:///'+ str (database_filepath))
    df = pd.read_sql ('SELECT * FROM Messages', engine)
    X = df ['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X,y,category_names


def tokenize(text):
    '''
    Processes the input text by cleaning, tokenizing, removing stopwords, 
    and applying emmatization.

    Input:
    - text: The raw text string that needs to be processed.

    Output:
    - lemmed: A list of cleaned and lemmatized tokens.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed
    


def build_model():
    '''
    Constructs a machine learning pipeline and performs hyperparameter tuning using GridSearchCV.

    The pipeline consists of the following steps:
    1. Text vectorization using CountVectorizer with a custom tokenizer.
    2. Transformation of the vectorized text into TF-IDF features.
    3. Multi-output classification using RandomForestClassifier.

    Input:
    - None (This function does not take any parameters directly).

    Output:
    - model: A GridSearchCV object that wraps the pipeline and allows for hyperparameter tuning.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [5],
        'clf__estimator__min_samples_split': [2],
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, verbose=2, cv=3)
    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the performance of a trained model on the test data.

    The function predicts the target labels for the test data and 
    then prints the classification report for each category.

    Input:
    - model: The trained machine learning model to evaluate.
    - X_test: The test data features (input data).
    - Y_test: The true labels for the test data (output data).
    - category_names: A list of the category names corresponding to the columns in Y_test.

    Output:
    - None (The function prints the evaluation metrics directly).
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Column: ",Y_test.columns[i])
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
    


def save_model(model, model_filepath):
    '''
    Saves the trained machine learning model to a file using the pickle format.

    Input:
    - model: The trained machine learning model that you want to save.
    - model_filepath: The file path where the model will be saved.

    Output:
    - None (The function saves the model to the specified file path).
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    '''
    The main function that orchestrates the entire machine learning pipeline:
    - Loads the data from a database.
    - Splits the data into training and test sets.
    - Builds, trains, and evaluates a machine learning model.
    - Saves the trained model to a file.

    The function expects two command-line arguments:
    1. The filepath of the SQLite database containing the disaster messages.
    2. The filepath where the trained model should be saved as a pickle file.

    Parameters:
    - None (The function relies on command-line arguments passed via sys.argv).

    Returns:
    - None (The function performs a series of actions and prints the status at each step).
    '''
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
