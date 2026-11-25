"""Read the classification notebook's Keras 3.8 HDF5 export."""
import json
from pathlib import Path

import h5py
import keras
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "source" / "Image_Detection_Model.h5"
CLASS_NAMES = ("Glioma", "Meningioma", "No Tumor", "Pituitary")
IMAGE_SIZE = (224, 224)


def load_classifier(path=MODEL_PATH):
    with h5py.File(path, "r") as artifact:
        model = json.loads(artifact.attrs["model_config"])
        writer = artifact.attrs["keras_version"]
    config = model["config"]
    if writer == "3.8.0" and model["class_name"] == "Sequential":
        nested = config["layers"][1]
        backbone = nested["config"]
        if (nested["class_name"] == "Functional"
                and backbone["name"] == "mobilenet_1.00_224"
                and backbone["input_layers"] == [["input_layer", 0, 0]]
                and backbone["output_layers"] == [["conv_pw_13_relu", 0, 0]]):
            # The training Sequential consumed one tensor. New readers preserve
            # the legacy singleton endpoint lists and pass a list to pooling.
            for key in ("input_layers", "output_layers"):
                backbone[key], = backbone[key]
            classifier = keras.Sequential.from_config(config)
            classifier.load_weights(path)
            return classifier
    return keras.models.load_model(path, compile=False)


def preprocess_image(image):
    rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    batch = np.asarray(rgb, dtype=np.float32)[None, ...]
    return batch * np.float32(1.0 / 255.0)


def classify(model, image):
    output = np.asarray(model(preprocess_image(image), training=False))
    if (output.shape != (1, len(CLASS_NAMES)) or not np.isfinite(output).all()
            or np.any(output < 0) or np.any(output > 1)
            or not np.isclose(output.sum(), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Unexpected four-class softmax output")
    return dict(zip(CLASS_NAMES, map(float, output[0]), strict=True))
