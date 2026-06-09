from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image


MODEL_PATH = Path("source") / "Image_Detection_Model.h5"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMAGE_SIZE = (224, 224)
DISCLAIMER = (
    "This project is for educational and portfolio demonstration only. "
    "It is not a medical diagnosis tool and should not be used for clinical decisions."
)

model = None
model_error = ""


def load_model():
    global model, model_error
    if model is not None or model_error:
        return model

    if not MODEL_PATH.exists():
        model_error = f"Model file not found: {MODEL_PATH}"
        return None

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        model_error = f"Could not load model: {exc}"
        model = None

    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_tumor(image: Image.Image):
    loaded_model = load_model()
    if loaded_model is None:
        return {"Model unavailable": 1.0}, model_error or "Model is unavailable."

    try:
        predictions = loaded_model.predict(preprocess_image(image), verbose=0)[0]
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim != 1:
            predictions = predictions.reshape(-1)

        class_count = min(len(CLASS_NAMES), len(predictions))
        scores = {
            CLASS_NAMES[index]: float(predictions[index])
            for index in range(class_count)
        }
        best_label = max(scores, key=scores.get)
        return scores, f"Top demo prediction: {best_label}. {DISCLAIMER}"
    except Exception as exc:
        return {"Prediction failed": 1.0}, f"Prediction failed: {exc}"


with gr.Blocks(title="Brain Tumor MRI AI Demo") as demo:
    gr.Markdown("# Brain Tumor MRI AI Demo")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(
        "Upload a non-sensitive sample MRI image to test the portfolio demo workflow. "
        "Outputs are model-demo signals, not medical advice."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Sample MRI image")
        with gr.Column():
            label_output = gr.Label(label="Prediction scores")
            note_output = gr.Textbox(label="Status", interactive=False)

    run_button = gr.Button("Run demo prediction")
    run_button.click(predict_tumor, inputs=image_input, outputs=[label_output, note_output])


if __name__ == "__main__":
    demo.launch()
