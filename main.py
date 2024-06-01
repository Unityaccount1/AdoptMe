from time import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jsonObjectImages=[]

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
    InputImage.id = len(jsonObjectImages) + 1
    jsonObjectImages.append(InputImage)
    message = "Imagen cargada correctamente"
    message = "Imagen cargada correctamente"
    jsonOutputMessage = {
        "message": message
    }
    return jsonOutputMessage
    
@app.post("/v1/getResponse")
async def uploadImage(InputConversation: InputJsonConversation):
    message = "Mostrando el total de imagenes cargadas"
    return jsonObjectImages
