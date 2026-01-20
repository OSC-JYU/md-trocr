from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import uuid
import json
from pydantic import BaseModel
from typing import List
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from PIL import Image
import numpy as np
import torch


from read_polygons import process_polygons

# Global variables to hold the model and processor
global_processor = None
global_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global global_processor
    global global_model
    print("Loading model...this may take a while...")
    try:
        global_processor, global_model = load_custom_trocr_model()
    except Exception as e:
        print(f"FATAL ERROR: Could not load the model on startup: {e}")
        exit(1)
    yield
    # Shutdown (if needed, add cleanup code here)

app = FastAPI(
    title="MessyDesk TrOCR API",
    description="API for TrOCR models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create necessary directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
POLYGON_BUFFER = 0.0
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Lazy loading - initialize as None, load on first use
processor = None
model = None



# code from https://huggingface.co/Kansallisarkisto/multicentury-htr-model
def load_custom_trocr_model():
    """Load a TrOCR model with custom image size support"""
    original_embeddings_forward = ViTEmbeddings.forward
    
    # Always apply patches for models saved with custom image sizes
    def universal_patch_forward(self, *args, **kwargs):
        pixel_values = args[0] if args else kwargs['pixel_values']
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
    def universal_embeddings_forward(self, *args, **kwargs):
        kwargs['interpolate_pos_encoding'] = True
        return original_embeddings_forward(self, *args, **kwargs)
    
    # Apply patches
    ViTPatchEmbeddings.forward = universal_patch_forward
    ViTEmbeddings.forward = universal_embeddings_forward
    
    # Load model and processor
    processor = TrOCRProcessor.from_pretrained("Kansallisarkisto/multicentury-htr-model",
                                               use_fast=True,
                                               do_resize=True, 
                                               size={'height': 192,'width': 1024})
     
    model = VisionEncoderDecoderModel.from_pretrained("Kansallisarkisto/multicentury-htr-model")
    print("TrOCR Model loaded successfully (2.23 GB) from Hugging Face.")

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to CUDA device.")
    else:
        print("CUDA not available. Using CPU.")    

    return processor, model

class ResponseData(BaseModel):
    type: str
    uri: List[str]

class ProcessResponse(BaseModel):
    response: ResponseData

@app.get("/")
async def root():
    return {"message": "TrOCR API for MessyDesk"}

@app.post("/process", response_model=ProcessResponse)
async def process_files(
    message: UploadFile = File(...),
    content: UploadFile = File(...),
    source: UploadFile = File(...)
):
    try:
        print("Processing files...")
        # Generate unique output ID
        output_id = str(uuid.uuid4())

        #  Check if model is loaded (should be from startup, but good practice)
        if global_model is None or global_processor is None:
            print("Model is not initialized. Server is starting up or failed to load model.")
            raise HTTPException(status_code=503, detail="Model is not initialized. Server is starting up or failed to load model.")

        # Save the uploaded files
        source_path = os.path.join(UPLOAD_FOLDER, f"{output_id}_source")
        with open(source_path, "wb") as f:
            f.write(await source.read())

        # Read and parse content as JSON
        content_data = await content.read()
        content_json = json.loads(content_data.decode('utf-8'))
        

        # read request as JSON
        message_data = await message.read()
        msg = json.loads(message_data.decode('utf-8'))

        polygons = content_json.get("line_polygons", [])

        ordered_polygons = process_polygons(
            polygons,
            Path(source_path),
            Path(OUTPUT_FOLDER),
            POLYGON_BUFFER,
        )

        # remove the original source file after processing
        try:
            os.remove(source_path)
        except Exception as cleanup_exc:
            print(f"Warning: could not remove original file {source_path}: {cleanup_exc}")


        # Use global model and processor
        processor = global_processor
        model = global_model
        lines = []
        
        for filename in ordered_polygons:   
            try:
                image = Image.open(filename).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                # move pixel_values to GPU if available
                if torch.cuda.is_available():
                    pixel_values = pixel_values.to("cuda")

                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                lines.append(generated_text)
                try:
                    os.remove(filename)
                except Exception as cleanup_exc:
                    print(f"Warning: could not remove temp file {filename}: {cleanup_exc}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                try:
                    os.remove(filename)
                except Exception as cleanup_exc:
                    print(f"Warning: could not remove temp file {filename}: {cleanup_exc}")


        with open(os.path.join(OUTPUT_FOLDER, f"{output_id}_text.txt"), "w") as f:
            f.write("\n".join(lines))

        return ProcessResponse(
            response=ResponseData(
                type="stored",
                uri=[f"/files/{output_id}_text.txt"]
            )
        )
        
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )



@app.get("/files/{filename}")
async def serve_file(filename: str, background_tasks: BackgroundTasks):

    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    def remove_file(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Error deleting file {path}: {e}")
            
    background_tasks.add_task(remove_file, file_path)
    return FileResponse(file_path, background=background_tasks)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012) 