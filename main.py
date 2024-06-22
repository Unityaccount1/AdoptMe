import os
import io
import base64

from time import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
#from langchain.embeddings import HuggingFaceEmbeddings
#from transformers import ViltProcessor, ViltForQuestionAnswering
#from PIL import Image

'''
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(imageBase64,text):
    try:
        image_bytes = base64.b64decode(imageBase64)
        img = Image.open(io.BytesIO(image)).convert("RGB")
        encoding = processor(img, text, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        
        return answer
    except Exception as e:
        return str(e)
#fileBase64: str = None
#description: str = None
#image_bytes = base64.b64decode(imagen)
#respuesta = image_bytes
'''
def conversionImagen(imagen,infoMascota,pregunta):
    try:
        arregloInfoMascota = infoMascota.split(" ")
        arregloPregunta = pregunta.split(" ")
        rep = 0
        for infoM in arregloInfoMascota:
            for infoP in arregloPregunta:
                if (infoP == infoM):
                    rep = rep + 1
        if (rep>=2) :
            respuesta = "success"
        else:
            respuesta = "error"
        return respuesta
    except Exception as e:
        return str(e)
        
class InputJsonConversation(BaseModel):
    id: Optional[int] = None
    message: str
    
 
    
class InputJsonImage(BaseModel):
    nombresApellidos: str
    correo: str
    direccion: str
    nombreMascota: str
    edadMascota: str
    mascotaVacuna: str
    mascotaEsteril: str
    razonAdopcion: str
    Foto: str
    descripcion: str
            

class OutputJsonImage(InputJsonImage):
    pass

class OutputJsonUploadImage(InputJsonConversation):
    pass
    

app = FastAPI()
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

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
    jsonObjectImages.append(InputImage)
    
    message = "Datos registrados correctamente"
    jsonOutputMessage = {
        "message": message
    }
    return jsonOutputMessage
    
@app.post("/v1/getResponse")
async def getResponse(InputConversation: InputJsonConversation):
    try:
        #answer = "Respuesta de prueba para verificar la conectividad"
        jsonConstructor = []
        a=0
        textoTotal = ""
        for index,test in enumerate(jsonObjectImages):
            textoDescripcion = test.mascotaEsteril + " " + test.mascotaVacuna + " " + test.nombreMascota + " " + test.edadMascota + " "+ test.razonAdopcion + test.descripcion
            valPrecision = conversionImagen(test.Foto,textoDescripcion,InputConversation.message)
            #validador: textoTotal = textoTotal + " " + valPrecision
            
            if ("success" in valPrecision):
                a=a+1
                jsonOutput = {
                    "id" : index,
                    "fileBase64" : test.Foto,
                    #"description" : Fototexto
                    "description" : "La mascota se llama: " + test.nombreMascota + " .Tiene: " + test.edadMascota + " años y la razon de su adopcion es:  "+ test.razonAdopcion
                    }
                jsonConstructor.append(jsonOutput)
        if (a>=1):    
            jsonOutputResponse = {
                "output" : jsonConstructor
            }
            return jsonOutputResponse
        else:
            jsonOutput = {
                    "id" : "0",
                    "fileBase64" : "",
                    #"description" : Fototexto
                    "description" : "No se encontraron registros."
                    }
            jsonConstructor.append(jsonOutput)
            jsonOutputResponse = {
                "output" : jsonConstructor
            }
            return jsonOutputResponse
        #validador
        #return InputConversation.message + " " + textoTotal
    except Exception as e:
        jsonConstructor = []
        jsonOutput = {
            "id" : "0",
            "fileBase64" : "",
            "description" : str(e)
            }
        jsonConstructor.append(jsonOutput)
        jsonOutputResponse = {
             "output" : jsonConstructor
         }
        return jsonOutputResponse
