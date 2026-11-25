"""Exercise the running Gradio demo and compare it with direct model outputs."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from gradio_client import Client, handle_file
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_model import CLASS_NAMES, MODEL_PATH, load_classifier, preprocess_image


def validate(url, sample):
    start = perf_counter()
    model = load_classifier()
    result = dict(load_seconds=perf_counter() - start, model=str(MODEL_PATH),
                  input_shape=model.input_shape, output_shape=model.output_shape,
                  parameters=model.count_params(), order=CLASS_NAMES, samples=[])
    client = Client(url, verbose=False)
    folder = MODEL_PATH.parents[1] / "dataset/Testing"
    paths = [sample] if sample else []
    if folder.exists():
        paths.extend(next(iter(sorted(d.glob("*.jpg"))), None)
                     for d in sorted(folder.iterdir()) if d.is_dir())
    paths = list(dict.fromkeys(p.resolve() for p in paths if p))
    if not paths:
        raise FileNotFoundError("Supply --sample with a real MRI image.")
    for path in paths:
        with Image.open(path) as image:
            batch = preprocess_image(image)
            size, mode = image.size, image.mode
        start = perf_counter()
        raw = model(batch, training=False).numpy()[0]
        elapsed = perf_counter() - start
        delta = max(float(np.max(np.abs(raw - model(batch, training=False).numpy()[0])))
                    for _ in range(3))
        assert delta <= 1e-6
        start = perf_counter()
        displayed, status = client.predict(handle_file(path), api_name="/predict_tumor")
        latency = perf_counter() - start
        scores = {v["label"]: v["confidence"] for v in displayed["confidences"]}
        np.testing.assert_allclose([scores[k] for k in CLASS_NAMES], raw, rtol=0, atol=1e-6)
        assert displayed["label"] == CLASS_NAMES[np.argmax(raw)] and displayed["label"] in status
        result["samples"].append(dict(file=str(path), size=size, mode=mode, batch=batch.shape,
                                      raw=raw.tolist(), top=displayed["label"], seconds=elapsed,
                                      api_seconds=latency, max_repeat_delta=delta, status=status))
    result["invalid"] = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, data in (("empty", None), ("corrupt.jpg", b"broken"), ("unsupported.svg", b"<svg/>")):
            upload = None
            if data:
                path = Path(directory) / name
                path.write_bytes(data)
                upload = handle_file(path)
            scores, status = client.predict(upload, api_name="/predict_tumor")
            assert not scores and status
            result["invalid"][name] = status
    final, _ = client.predict(handle_file(paths[0]), api_name="/predict_tumor")
    assert final["label"] == result["samples"][0]["top"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--sample", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.url, args.sample), indent=2))
