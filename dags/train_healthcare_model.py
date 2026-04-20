from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def retrain_model():
    # Run your training script
    subprocess.run(["uv", "run", "python", "src/train_model.py"], check=True)

with DAG(
    dag_id="train_healthcare_model",
    start_date=datetime(2026, 4, 20),
    schedule_interval="0 12 * * SAT",  # every Saturday at noon
    catchup=False,
) as dag:

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )
