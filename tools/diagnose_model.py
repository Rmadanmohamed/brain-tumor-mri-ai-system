"""Probe the original HDF5 graph, then exercise Keras' unmodified loader."""
import json
from pathlib import Path

import h5py
import keras
from keras.src.legacy.saving import saving_utils

path = Path(__file__).resolve().parents[1] / "source" / "Image_Detection_Model.h5"
with h5py.File(path) as artifact:
    layers = json.loads(artifact.attrs["model_config"])["config"]["layers"]
    print("Writer:", artifact.attrs["keras_version"], "Reader:", keras.__version__)
    for key in ("input_layers", "output_layers"):
        print(key, layers[1]["config"][key])
backbone = saving_utils.model_from_config(layers[1])
features = backbone(keras.Input((224, 224, 3)))
pool = saving_utils.model_from_config(layers[2])
print("Symbolic backbone output:", features)
print("Symbolic pooling output:", pool(features), flush=True)
model = keras.models.load_model(path, compile=False)
print(model.input_shape, model.output_shape, model.count_params())
