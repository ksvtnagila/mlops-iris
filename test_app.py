from fastapi.testclient import TestClient
from main import app


def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ping":"pong"}

def test_pred_virginica():
    payload = {
      "sepal_length": 6.3,
      "sepal_width": 3.3,
      "petal_length": 6.0,
      "petal_width": 2.5
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Virginica"}
        
def test_pred_setosa():
    payload = {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Setosa"}
        
def test_pred_versicolor():
    payload = {
      "sepal_length": 7.0,
      "sepal_width": 3.2,
      "petal_length": 4.7,
      "petal_width": 1.4
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Versicolour"}