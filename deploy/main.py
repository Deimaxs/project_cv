from fastapi import FastAPI, Body
from pydantic import BaseModel
from prediction import Model
import base64

app = FastAPI()

class Video(BaseModel):
    video: str
    skip: int
    threshold: float
    function: str
    object_vis: str

@app.post("/")
def predict_request(video: Video = Body(...)):
    request_data = video.dict()
    example = Model()
    response_64 = example.predict(request_data["video"], request_data["skip"], request_data["threshold"], request_data["function"], request_data["object_vis"])
    return {"video": response_64}
