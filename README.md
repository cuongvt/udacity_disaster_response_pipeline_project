# Disaster Response Pipeline Project
This repository contains the source code for a Flask web application and ETL/ML pipeline that analyzes disaster data from Appen to build a model for an API that classifies disaster messages.

**Repository Structure:**

* **app/:** Flask web application which can get user input message and classify to approximate categories.
* **data/:** Stores the raw, database and ETL process.
* **models/:** Stores the trained model, pipeline for train model
* **README.md:** This file.

**Getting Started:**

1. **Clone the repository:** 
```bash
   git clone https://github.com/cuongvt/udacity_disaster_response_pipeline_project/tree/main
   https://github.com/cuongvt/stack-over-flow-insight-udacity-example/tree/main
```

2. **Run the following commands in the project's root directory to set up your database and model.**

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. **Run the following command in the app's directory to run your web app.**
    `python run.py`

4. **Go to** http://0.0.0.0:3001/
