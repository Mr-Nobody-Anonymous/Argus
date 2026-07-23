"""
Model Optimization for Production Surveillance
Provides TensorRT/ONNX optimizations for different hardware
"""
import os
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimizes AI models for production deployment.
    
    Hardware Recommendations:
    - NVIDIA Jetson: ONNX Runtime with CUDA, INT8 quantization
    - NVIDIA T4/V100: TensorRT FP16 optimization
    - CPU-only: ONNX Runtime with OpenMP, OpenVINO
    - Mixed: Dynamic batching with TorchScript
    """

    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_mode = self._detect_optimal_mode()

    def _detect_optimal_mode(self) -> str:
        """Detect optimal optimization based on hardware"""
        if "Jetson" in os.uname().release if hasattr(os, 'uname') else "":
            return "onnx_int8"  # Jetson Nano/Xavier
        elif torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 16:  # T4/V100/A100
                return "tensorrt_fp16"
            else:  # Smaller GPU
                return "tensorrt_int8"
        return "onnx_cpu"

    def export_to_onnx(self, model_path: str = "yolov8s.pt", output_path: str = "models/yolov8s.onnx"):
        """
        Export YOLO model to ONNX for cross-platform deployment.
        ONNX provides 2-3x speedup on CPU and GPU.
        """
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # Export with optimizations
            success = model.export(
                format="onnx",
                imgsz=640,
                half=True,  # FP16
                simplify=True,  # Simplify graph
                opset=12
            )
            
            if success:
                logger.info(f"Exported ONNX model to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            
        return None

    def convert_to_tensorrt(self, onnx_path: str, output_path: str = "models/yolov8s.trt"):
        """
        Convert ONNX to TensorRT for NVIDIA GPU optimization.
        Provides 4-8x speedup on NVIDIA hardware.
        """
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.BATCH))
            
            # Parse ONNX
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
            
            # Build optimized engine
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB workspace
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
            
            engine = builder.build_engine(network, config)
            
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
                
            logger.info(f"Built TensorRT engine: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("TensorRT not available, skipping conversion")
            return None

    def optimize_for_inference(self, model_path: str) -> str:
        """
        Main optimization entry point.
        Returns path to optimized model.
        """
        if self.optimization_mode == "tensorrt_fp16":
            onnx_path = model_path.replace('.pt', '.onnx')
            if not os.path.exists(onnx_path):
                self.export_to_onnx(model_path, onnx_path)
            trt_path = onnx_path.replace('.onnx', '.trt')
            if os.path.exists(onnx_path):
                return self.convert_to_tensorrt(onnx_path, trt_path) or onnx_path
        else:
            onnx_path = model_path.replace('.pt', '.onnx')
            if not os.path.exists(onnx_path):
                return self.export_to_onnx(model_path, onnx_path)
            return onnx_path
        
        return model_path


# Inference with ONNX Runtime
class OptimizedInference:
    """
    ONNX Runtime inference for maximum performance.
    Supports CUDA, TensorRT, and CPU execution providers.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self._load_model()

    def _load_model(self):
        """Load optimized model"""
        try:
            import onnxruntime as ort
            
            providers = []
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"Loaded ONNX model with {providers}")
            
        except ImportError:
            logger.warning("ONNX Runtime not available")

    def infer(self, frame):
        """Run inference on frame"""
        if self.session is None:
            return None
        
        # Preprocess frame
        input_name = self.session.get_inputs()[0].name
        input_tensor = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        return self._postprocess(outputs)

    def _preprocess(self, frame):
        """Preprocess frame for ONNX model"""
        import numpy as np
        
        # Resize and normalize
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0
        return img

    def _postprocess(self, outputs):
        """Post-process model outputs"""
        # Decode YOLO outputs
        return outputs


def get_optimization_guide():
    """Return hardware-specific optimization guide"""
    return """
    # Model Optimization Guide

    ## NVIDIA Jetson (Nano/Xavier):
    - Use ONNX Runtime with CUDA
    - INT8 quantization reduces model size by 4x
    - Enable --half flag for FP16 where supported

    ## NVIDIA T4/V100/A100:
    - Convert to TensorRT FP16
    - Dynamic batching for multiple cameras
    - Expected: 4-8x speedup, 50+ FPS per camera

    ## CPU-only (Intel/AMD):
    - ONNX Runtime with OpenMP
    - OpenVINO for Intel CPUs
    - Reduce inference size to 416x416

    ## Memory Optimization Tips:
    - Use half() for FP16 models
    - Enable torch.compile() for PyTorch 2.0+
    - Batch multiple camera frames for inference
    """