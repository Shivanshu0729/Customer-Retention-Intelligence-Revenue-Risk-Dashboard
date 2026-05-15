import os

def test_project_files():

    assert os.path.exists("app.py")

    assert os.path.exists("Churn_Modelling.csv")

    assert os.path.exists("Dockerfile")