import logging
import os

import gradio as gr
from PIL import Image

DISCLAIMER = (
    "This project is for educational and portfolio demonstration only. "
    "It is not a medical diagnosis tool and should not be used for clinical decisions."
)
logger = logging.getLogger(__name__)
UNAVAILABLE = "Model unavailable. Check the server log and restart after correcting the model setup."


def initialize_model():
    try:
        from mri_model import load_classifier
        return load_classifier()
    except Exception:
        logger.exception("Classifier initialization failed")
        return None


model = initialize_model()


def predict_tumor(image):
    if model is None:
        return None, UNAVAILABLE
    if image is None:
        return None, "Upload a sample MRI image first."
    try:
        from mri_model import classify
        scores = classify(model, image)
        return scores, f"Top demo prediction: {max(scores, key=scores.get)}. {DISCLAIMER}"
    except Exception:
        logger.exception("Classifier request failed")
        return None, "Prediction failed. Check the server log."


def predict_upload(upload):
    if not upload:
        return predict_tumor(None)
    # Gradio validates and moves uploads into its cache before passing the
    # serialized ImageData here. Decode inside the event to clear stale scores.
    try:
        with Image.open(upload["path"]) as image:
            image.load()
            return predict_tumor(image)
    except (OSError, ValueError, KeyError, Image.DecompressionBombError):
        logger.exception("Image decoding failed")
        return None, "Could not read this image. Upload a readable raster image such as JPEG or PNG."


with gr.Blocks(title="Brain Tumor MRI AI Demo", analytics_enabled=False) as demo:
    gr.Markdown("# Brain Tumor MRI AI Demo")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(
        "Upload a non-sensitive sample MRI image to test the portfolio demo workflow. "
        "The classifier also assigns scores to unrelated images."
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Sample MRI image", format="png")
        with gr.Column():
            label_output = gr.Label(label="Prediction scores")
            note_output = gr.Textbox(label="Status", interactive=False,
                                     value="" if model is not None else UNAVAILABLE)
    run_button = gr.Button("Run demo prediction", interactive=model is not None)
    run_button.click(predict_upload, inputs=image_input, outputs=[label_output, note_output],
                     preprocess=False, api_name="predict_tumor")


if __name__ == "__main__":
    import uvicorn
    from web_server import create_server

    api = create_server(model)
    if os.getenv("MRI_CLASSIC", "1") == "1":
        api = gr.mount_gradio_app(api, demo, path="/classic", max_file_size="10mb")
    uvicorn.run(api, host=os.getenv("MRI_HOST", "127.0.0.1"),
                port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
