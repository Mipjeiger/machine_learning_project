from prefect import task, flow
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

@task
def load_data(path: str):
    df = pd.read_csv(path)
    return df

@task
def preprocess_data(df: pd.DataFrame):
    # ...(Logika Preprocessing)
    X = df.drop(['fraud'], axis=1)
    y = df['fraud']
    return X, y

@task
def train_model(X, y):
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric("accuracy", accuracy)

        # simpan model ke file dill 
        with open('ml_prefect_analysis_fraud.dill', 'wb') as f:
            dill.dump(model ,f)
        
    return 'ml_prefect_analysis_fraud.dill'

@flow
def fraud_detection_pipeline(data_path: str):
    raw_data = load_data(data_path)
    X, y = preprocess_data(raw_data)
    model_file = train_model(X, y)
    print(f"Model baru disimpan sebagai: {model_file}")

if __name__ == "__main__":
    fraud_detection_pipeline(data_path="Machine_learning_fraud.csv")