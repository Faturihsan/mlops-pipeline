from fastapi import FastAPI, Request
import onnxruntime as ort
import numpy as np
from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    input: List[List[float]]

app = FastAPI()
session = ort.InferenceSession("../shared_models/model.onnx")

@app.post("/predict")
async def predict(data: InputData):
    input_array = np.array(data.input, dtype=np.float32)
    result = session.run(None, {"input": input_array})[0]
    return {"prediction": result.tolist()}
