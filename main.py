import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict, load_model2, predict2
from datetime import datetime

app = FastAPI(
    title="Iris Predictor",
    docs_url="/"
)

app.add_event_handler("startup", load_model2)

class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class QueryOut(BaseModel):
    flower_class: str


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict_flower", response_model=QueryOut, status_code=200)
def predict_flower(
    query_data: QueryIn
):
    timeNow = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    output = {'flower_class': predict2(query_data), 'timeStamp': timeNow}
    return output

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8887, reload=True)
