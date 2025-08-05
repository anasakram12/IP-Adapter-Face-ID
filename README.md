
# IP-Adapter Face ID Image Generation

This project uses **Stable Diffusion** combined with **InsightFace Buffalo\_L** for **face-conditioned image generation**. It generates high-quality images that retain facial identity while following text prompts.

---

## 🖼️ Features

* Face embedding extraction with **InsightFace Buffalo\_L**.
* IP-Adapter FaceID integration for precise face-conditioned outputs.
* High-quality image generation at **512x768 resolution**.
* Based on **Realistic Vision V4.0** Stable Diffusion model.
* Negative prompt support for output quality control.
* Batch generation of 4 images per run.

---

## 📦 Installation

Ensure the following Python packages are installed:

```bash
pip install insightface diffusers transformers accelerate
pip install mediapipe onnxruntime torch torchvision opencv-python matplotlib
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
```

---

## 🚀 How to Run

1. Clone this repository and navigate to the project directory.
2. Update the **input image URL or path** inside `Image_Gen.py`:

   ```python
   image_url = "https://your-image-url.com/image.jpg"
   ```
3. Run the script:

   ```bash
   python Image_Gen.py
   ```
4. The generated image grid will be saved as:

   ```
   /kaggle/working/generated_grid.png
   ```

---

## 🖥️ Requirements

| Component | Specification                      |
| --------- | ---------------------------------- |
| GPU       | CUDA-compatible (8GB VRAM minimum) |
| Framework | PyTorch with CUDA support          |
| Platform  | Kaggle / Google Colab / Local GPU  |

---

## 📁 Project Structure

```
Image_Gen.py          # Main script for face-conditioned image generation
models/               # Auto-downloaded models (from Hugging Face Hub)
generated_grid.png    # Output image grid (saved after generation)
```

---

## 🔑 Key Parameters (Inside Code)

| Parameter         | Value                      |
| ----------------- | -------------------------- |
| Prompt            | Customizable text prompt   |
| Negative Prompt   | To avoid artifacts         |
| Inference Steps   | 30                         |
| Output Resolution | 512x768 pixels             |
| Batch Size        | 4 images per run           |
| Seed              | 2023 (for reproducibility) |

---

## 📜 License

MIT License.

---

