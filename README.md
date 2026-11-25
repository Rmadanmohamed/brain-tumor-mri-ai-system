# NeuroScope — browser deployment

Static browser-only build derived from source commit `6780fbd6c055e9fba72c87d0a59386ea42ab1574`. The original Python app and trained HDF5 file remain on `main`. This Pages tree contains only the original visual frontend, a converted ONNX export of the same loaded weights, and the pinned ONNX Runtime Web WASM CPU runtime.

Inference runs locally in your browser. Uploaded images are not sent to a server. This educational portfolio demonstration is not a clinically validated medical diagnosis system.

The project owner authorized public redistribution of the converted model and visual assets. See included license notices for the project, fonts/icons, and ONNX Runtime Web.
