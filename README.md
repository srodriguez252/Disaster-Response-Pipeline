# Disaster Response Pipeline Project

Machine learning pipeline that categorizes emergency messages based on the needs communicated by the sender. The project is divided into 3 parts: an ETL pipeline that processes the data and creates a database, an ML pipeline that trains a machine learning model with the database provided by the ETL pipeline, and a web application that consumes the database and the ML model to classify emergency messages.



## Instalation and Setup

1. Clone the repository:
```sh
git clone https://github.com/srodriguez252/Disaster-Response-Pipeline.git
```

2. Install the dependencies:
```sh
pip install -r requirements.txt
```

## Executing Program
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
     ```sh
       `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     ```
    - To run ML pipeline that trains classifier and saves
    ```sh
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    ```
2. Run the following command in the app's directory to run your web app.
   ```sh
     `python run.py`
   ```
4. Go to http://0.0.0.0:3001/

## Aditional Considerations
The dataset of this project is unbalanced, meaning that some columns have fewer examples compared to the others. This imbalance is common in real-world datasets where some categories are not as common as others.
Inbalance in datasets can lead to a model that is biased towards the majority classes, potentially neglecting the minority of classes. During training, the model might learn to predict the most frequent categories, underperforming on the less frequent ones.

In order to address this imbalance in the training process, some strategies could be used. Adjusting class weights to give more importance to the minority classes can help the model pay more attention to those categories. Instead of focusing on accuracy, which can be misleading in imbalanced datasets, metrics like f-1 can be used to evaluate the model's performance.
