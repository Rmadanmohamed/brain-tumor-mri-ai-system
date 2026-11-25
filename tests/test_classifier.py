import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import keras
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import app
from mri_model import CLASS_NAMES, MODEL_PATH, classify, load_classifier, preprocess_image

SAMPLE = MODEL_PATH.parents[1] / "dataset/Testing/glioma/Te-gl_1.jpg"


class ClassifierTests(unittest.TestCase):
    def test_graph_and_every_weight(self):
        model = app.model
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 4))
        self.assertEqual(model.count_params(), 3524676)
        with h5py.File(MODEL_PATH) as source:
            for layer in model.layers:
                group = source["model_weights"][layer.name]
                weights = {name.removeprefix("sequential/"): group[name]
                           for name in group.attrs["weight_names"]}
                for variable in layer.weights:
                    saved = weights.pop(variable.path.removeprefix("sequential/"))
                    np.testing.assert_array_equal(variable.numpy(), saved[()])
                self.assertFalse(weights)

    @unittest.skipUnless(SAMPLE.is_file(), "Local MRI dataset unavailable")
    def test_training_preprocessing_and_repeatability(self):
        expected = keras.utils.img_to_array(keras.utils.load_img(SAMPLE, target_size=(224, 224)))
        expected = ImageDataGenerator(rescale=1./255).standardize(expected)
        with Image.open(SAMPLE) as image:
            np.testing.assert_array_equal(preprocess_image(image)[0], expected)
            scores = classify(app.model, image)
            self.assertEqual(tuple(scores), CLASS_NAMES)
            for _ in range(3):
                np.testing.assert_allclose(list(classify(app.model, image).values()),
                                           list(scores.values()), rtol=0, atol=1e-6)

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_classifier(Path(folder) / "missing.h5")
            with patch("mri_model.load_classifier", side_effect=FileNotFoundError):
                with self.assertLogs("app", level="ERROR"):
                    self.assertIsNone(app.initialize_model())
        with patch.object(app, "model", None):
            self.assertEqual(app.predict_tumor(None), (None, app.UNAVAILABLE))

    def test_invalid_input(self):
        self.assertIsNone(app.predict_upload(None)[0])
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "bad.jpg"
            broken.write_bytes(b"invalid")
            with self.assertLogs("app", level="ERROR"):
                scores, status = app.predict_upload({"path": str(broken)})
            self.assertIsNone(scores)
            self.assertNotIn(folder, status)

    def test_grayscale_rgb_and_unrelated_image(self):
        for mode in ("RGB", "L"):
            with self.subTest(mode=mode):
                scores, _ = app.predict_tumor(Image.new(mode, (1, 19)))
                self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)
