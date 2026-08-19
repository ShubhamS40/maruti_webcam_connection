"""
CNN inference: 224x224 RGB, [0,1] normalize, multi-output handling, temporal smoothing.
"""

import os
from collections import deque

import cv2
import numpy as np

from config import BASE_DIR, CNN_INPUT_SIZE, CNN_SMOOTH_WINDOW

TFLITE_PATH = os.path.join(BASE_DIR, "models", "drowsy_model.tflite")
H5_PATH = os.path.join(BASE_DIR, "models", "drowsy_model_full.h5")


class CnnPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self._interpreter = None
        self._input_index = None
        self._output_index = None
        self._backend = "none"
        self.cnn_history = deque(maxlen=40)
        self.raw_score_val = 0.0
        self._load(model_path or H5_PATH)

    def _load(self, model_path):
        if os.path.isfile(model_path):
            try:
                from tensorflow.keras.models import load_model

                self.model = load_model(model_path, compile=False)
                self._backend = "keras"
                print(f"CNN loaded (Keras): {model_path}")
                return
            except Exception as exc:
                print(f"Keras load failed: {exc}")

        if os.path.isfile(TFLITE_PATH):
            try:
                import tensorflow as tf

                self._interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
                self._interpreter.allocate_tensors()
                self._input_index = self._interpreter.get_input_details()[0]["index"]
                self._output_index = self._interpreter.get_output_details()[0]["index"]
                self._backend = "tflite"
                print(f"CNN loaded (TFLite): {TFLITE_PATH}")
                return
            except Exception as exc:
                print(f"TFLite load failed: {exc}")

        print("WARNING: CNN unavailable — drowsiness CNN term will be 0.")

    @staticmethod
    def _parse_output(out):
        flat = np.asarray(out).reshape(-1)
        if flat.size == 0:
            return 0.0
        if flat.size == 1:
            return float(flat[0])
        if flat.size == 2:
            a, b = float(flat[0]), float(flat[1])
            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and abs((a + b) - 1.0) < 0.15:
                return b
            return float(np.max(flat))
        return float(np.mean(flat))

    def preprocess(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None
        face = cv2.resize(face_bgr, CNN_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # MobileNetV2 preprocessing: scale from [0, 255] to [-1, 1] range
        face_scaled = (face.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(face_scaled, axis=0)

    def _infer(self, tensor):
        if self._backend == "keras" and self.model is not None:
            return self._parse_output(self.model.predict(tensor, verbose=0))
        if self._backend == "tflite" and self._interpreter is not None:
            self._interpreter.set_tensor(self._input_index, tensor.astype(np.float32))
            self._interpreter.invoke()
            return self._parse_output(
                self._interpreter.get_tensor(self._output_index)
            )
        return 0.0

    def predict(self, face_bgr):
        tensor = self.preprocess(face_bgr)
        if tensor is None:
            return self.smoothed_score, self.raw_score

        try:
            raw = max(0.0, min(1.0, self._infer(tensor)))
        except Exception:
            raw = 0.0

        self.raw_score_val = raw
        self.cnn_history.append(raw)
        return self.smoothed_score, raw

    @property
    def smoothed_score(self):
        if not self.cnn_history:
            return 0.0
        return float(sum(self.cnn_history) / len(self.cnn_history))

    @property
    def raw_score(self):
        return self.raw_score_val

