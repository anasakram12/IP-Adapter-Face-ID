#--------------------------------------------------------------------------
# INSTALL REQUIRED PACKAGES
#--------------------------------------------------------------------------
# Uncomment and run the following lines if you don't have these packages installed.
# !pip install insightface
# !pip install diffusers transformers accelerate
# !pip install mediapipe
# !pip install onnxruntime
# !pip install torch torchvision
# !pip install opencv-python
# !pip install matplotlib
# !pip install git+https://github.com/tencent-ailab/IP-Adapter.git

#--------------------------------------------------------------------------
# IMPORT NECESSARY LIBRARIES
#--------------------------------------------------------------------------
import os
import cv2
import torch
import requests
from PIL import Image
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

#--------------------------------------------------------------------------
# SETUP: DOWNLOAD MODELS AND PREPARE ENVIRONMENT
#--------------------------------------------------------------------------
# Create a directory to store the downloaded models.
os.makedirs('/kaggle/working/models', exist_ok=True)

# Download the IP-Adapter FaceID model from Hugging Face Hub.
print("Downloading IP-Adapter face model...")
ip_ckpt_path = hf_hub_download(
    repo_id="h94/IP-Adapter-FaceID",
    filename="ip-adapter-faceid_sd15.bin",
    local_dir="/kaggle/working"
)

# Download the image encoder configuration and model files.
print("Downloading image encoder...")
hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/image_encoder/config.json",
    local_dir="/kaggle/working/models/image_encoder" # Corrected directory
)
hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/image_encoder/pytorch_model.bin",
    local_dir="/kaggle/working/models/image_encoder" # Corrected directory
)

#--------------------------------------------------------------------------
# USER INPUT: PROVIDE YOUR IMAGE
#--------------------------------------------------------------------------
# Provide the path to your input image here.
# You can upload your own image and update the path.
# For example: input_image_path = "path/to/your/image.jpg"

# As an example, we are downloading an image from a URL.
# Replace this with the path to your local image file.
image_url = "https://pbs.twimg.com/profile_images/541867053351583744/rcxem8NU_400x400.jpeg"
input_image_path = "my_face_image.jpg"

response = requests.get(image_url)
if response.status_code == 200:
    with open(input_image_path, "wb") as f:
        f.write(response.content)
    print(f"Image downloaded and saved as '{input_image_path}'")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

#--------------------------------------------------------------------------
# MODEL CONFIGURATION
#--------------------------------------------------------------------------
# Define the paths for the models.
# These models are downloaded from Hugging Face Hub.
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/kaggle/working/models/image_encoder/"
ip_ckpt = "/kaggle/working/ip-adapter-faceid_sd15.bin"
device = "cuda" # Use "cuda" for GPU or "cpu" for CPU.

#--------------------------------------------------------------------------
# FACE ANALYSIS AND EMBEDDING EXTRACTION
#--------------------------------------------------------------------------
# Initialize the FaceAnalysis model to detect faces and extract embeddings.
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load the input image and get face embeddings.
# This will be used to guide the image generation.
image = cv2.imread(input_image_path)
faces = app.get(image)

if not faces:
    raise ValueError("No face detected in the provided image.")

# Convert the face embedding to a PyTorch tensor.
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

#--------------------------------------------------------------------------
# SETUP STABLE DIFFUSION PIPELINE
#--------------------------------------------------------------------------
# Configure the noise scheduler for the diffusion process.
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# Load the VAE (Variational Autoencoder).
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# Load the main Stable Diffusion pipeline.
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

#--------------------------------------------------------------------------
# IMAGE GENERATION
#--------------------------------------------------------------------------
# Load the IP-Adapter FaceID model into the pipeline.
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

# Define the prompt and negative prompt for image generation.
# The prompt describes the desired output image.
# The negative prompt lists elements to avoid in the image.
prompt = "photo of a 5 year old luaghing gir kid in Story book cover page"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

# Generate the images using the IP-Adapter model.
# This combines the text prompt with the face embedding.
print("Generating images...")
images = ip_model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    faceid_embeds=faceid_embeds,
    num_samples=4,
    width=512,
    height=768,
    num_inference_steps=30,
    seed=2023
)

#--------------------------------------------------------------------------
# SAVE AND DISPLAY RESULTS
#--------------------------------------------------------------------------
# This is a helper function to create a grid of images.
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Create a grid from the generated images.
# Change the rows and columns based on the number of samples.
grid = image_grid(images, 1, 4)

# Define the output path for the generated image grid.
output_image_path = '/kaggle/working/generated_grid.png'
grid.save(output_image_path)

print(f"Generation complete! Image grid saved to {output_image_path}")

# To display the image in a Jupyter Notebook or similar environment,
# you can simply call the 'grid' variable at the end.
# grid