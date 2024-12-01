import os
import io
import base64
from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from google.cloud import storage
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Define GCS bucket and model path
BUCKET_NAME = "text-to-image-model-429503"
MODEL_FOLDER = "model_artifacts/model"  # Folder in the bucket containing the model

def download_model_from_gcs(bucket_name, model_folder, local_model_path):
    """Downloads the model from a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_folder)
    
    for blob in blobs:
        # Create local directories if necessary
        local_file_path = os.path.join(local_model_path, os.path.relpath(blob.name, model_folder))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        blob.download_to_filename(local_file_path)
        print(f"Downloaded: {blob.name} to {local_file_path}")

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
import torch
import torch_xla
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
t1 = torch.ones(3, 3, device=dev)
print(t1)

# Download model to local directory if not already present
local_model_path = "local_model"
if not os.path.exists(local_model_path):
    print("Downloading model from GCS...")
    download_model_from_gcs(BUCKET_NAME, MODEL_FOLDER, local_model_path)
else:
    print("Model already downloaded locally.")

# Load the Stable Diffusion model
model_path = local_model_path
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

# Check if CUDA is available; fall back to CPU if not
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe.to(device)

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        # Parse the JSON payload
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Generate the image
        image = pipe(prompt=prompt).images[0]

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the base64 string
        return jsonify({"image": img_base64})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
