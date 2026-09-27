"""
Unified Detector Module for Helmet & No-Helmet Detection
Supports:
1. PyTorch YOLOv8 (.pt)
2. Standard ONNX (.onnx)
3. Quantized ONNX INT8 (.onnx)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import os
import cv2
import numpy as np


@dataclass
class Detection:
    """Represents a single detected bounding box with class and confidence."""
    xyxy: np.ndarray  # [x1, y1, x2, y2] in original image coordinates
    confidence: float
    class_id: int     # 0: helmet, 1: no_helmet
    class_name: str

    @property
    def box_xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


class BaseDetector(ABC):
    """Abstract Base Class for Helmet Detectors."""
    
    CLASSES = {0: "helmet", 1: "no_helmet"}

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50
    ) -> List[Detection]:
        """Detect objects in a BGR OpenCV image."""
        pass


class YOLOv8PyTorchDetector(BaseDetector):
    """YOLOv8 PyTorch (.pt) Inference Engine using Ultralytics."""

    def __init__(self, model_path: str, device: str = "cpu"):
        from ultralytics import YOLO
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        self.model = YOLO(model_path)
        self.device = device

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50
    ) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            # Map class name from model names dictionary if available
            cls_name = self.model.names.get(cls_id, self.CLASSES.get(cls_id, f"class_{cls_id}"))
            # Normalize class names to standard keys
            normalized_name = "helmet" if "helmet" in cls_name.lower() and "no" not in cls_name.lower() else "no_helmet"
            normalized_id = 0 if normalized_name == "helmet" else 1

            detections.append(Detection(
                xyxy=box.astype(np.float32),
                confidence=float(conf),
                class_id=normalized_id,
                class_name=normalized_name
            ))
        return detections


class ONNXDetector(BaseDetector):
    """
    High-performance ONNX Runtime Engine.
    Executes both FP32 and Quantized INT8 ONNX models with CPU optimizations.
    """

    def __init__(self, model_path: str, input_size: int = 640, num_threads: int = 4):
        import onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        self.input_size = input_size
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Vectorized Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50
    ) -> List[Detection]:
        orig_h, orig_w = image.shape[:2]
        img_padded, ratio, (dw, dh) = self._letterbox(image, (self.input_size, self.input_size))

        # Preprocess: BGR -> RGB, float32, normalize [0, 1], HWC -> CHW -> NCHW
        blob = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # ONNX Runtime Inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        output = outputs[0]  # Shape: [1, 4 + num_classes, num_anchors] e.g. [1, 6, 8400]

        predictions = np.squeeze(output, axis=0).T  # Transposed to [8400, 6]
        boxes = predictions[:, :4]  # [cx, cy, w, h]
        class_scores = predictions[:, 4:]  # class probabilities

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        valid_mask = max_scores >= conf_threshold

        boxes = boxes[valid_mask]
        scores = max_scores[valid_mask]
        class_ids = class_ids[valid_mask]

        if len(boxes) == 0:
            return []

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Scale coordinates back to original frame dimensions
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / ratio
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

        # NMS per class
        detections: List[Detection] = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            keep_indices = self._nms(cls_boxes, cls_scores, iou_threshold)
            cls_name = self.CLASSES.get(int(cls), f"class_{cls}")

            for idx in keep_indices:
                detections.append(Detection(
                    xyxy=cls_boxes[idx].astype(np.float32),
                    confidence=float(cls_scores[idx]),
                    class_id=int(cls),
                    class_name=cls_name
                ))
        return detections


def create_detector(model_path: str, device: str = "cpu") -> BaseDetector:
    """Factory function to instantiate detector based on file extension."""
    if model_path.endswith(".pt"):
        return YOLOv8PyTorchDetector(model_path, device=device)
    elif model_path.endswith(".onnx"):
        return ONNXDetector(model_path)
    else:
        raise ValueError(f"Unsupported model extension: {model_path}. Use .pt or .onnx")
