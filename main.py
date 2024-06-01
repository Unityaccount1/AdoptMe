import os

from time import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


class InputJsonConversation(BaseModel):
    id: Optional[int] = None
    message: str = None
    
    
class InputJsonImage(BaseModel):
    id: Optional[int] = None
    fileBase64: str
    description: str
            

class OutputJsonImage(InputJsonImage):
    pass

class OutputJsonUploadImage(InputJsonConversation):
    pass
    

app = FastAPI()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jsonObjectImages=[]
LongChainText = ""

@app.middleware("http")
async def log_middleware(request, call_next):
    start_time = time()
    response = await call_next(request)
    end_time = time()
    process_time = end_time - start_time
    print(f"Request {request.url} proccesed in {process_time} in sec")
    return response

@app.post("/v1/uploadImage")
async def uploadImage(InputImage: InputJsonImage):
    text = ""
    InputImage.id = len(jsonObjectImages) + 1
    text = text + InputImage.description
    LongChainText = text
    jsonObjectImages.append(InputImage)
    
    message = "Imagen cargada correctamente"
    jsonOutputMessage = {
        "message": message
    }
    return jsonOutputMessage
    
@app.post("/v1/getResponse")
async def getResponse(InputConversation: InputJsonConversation):
    embeddings = HuggingFaceEmbeddings()
    question_answering = pipeline("question-answering")
    extracted_text = LongChainText
    if InputConversation.message:
        answer = question_answering(question=InputConversation.message, context=extracted_text)
    jsonConstructor = []
    jsonOutput = {
        "id" : "1",
        "fileBase64" : "",
        "description" : answer
        }
    jsonConstructor.append(jsonOutput)
    jsonOutputResponse = {
        "output" : jsonConstructor
        }
    return jsonOutputResponse
    
    
