"""OCR model manager supporting DeepSeek OCR model."""

import sys
from io import StringIO
from logger import logger

# Try to import DeepSeek dependencies (optional)
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    torch = None
    AutoModel = None
    AutoTokenizer = None

# Base path for shared model weights on the UKP cluster
SHARED_MODELS_BASE = "/storage/ukp/shared/shared_model_weights"

# Configuration for DeepSeek OCR model
OCR_MODEL_CONFIG = {
    "deepseek-ocr": {
        "type": "deepseek",
        "path": f"{SHARED_MODELS_BASE}/models--deepseek-ai--DeepSeek-OCR",
        "description": "DeepSeek OCR - Vision-language model for OCR and table detection with grounding",
    }
}


class OCRModelManager:
    """Manager for DeepSeek OCR model."""
    
    def __init__(self):
        self.model_name = None
        self.model_type = None
        
        # DeepSeek-specific attributes
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_name: str = "deepseek-ocr") -> bool:
        """
        Load DeepSeek OCR model.
        
        Args:
            model_name: Name of the OCR model to load ("deepseek-ocr")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in OCR_MODEL_CONFIG:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(OCR_MODEL_CONFIG.keys())}")
            
            config = OCR_MODEL_CONFIG[model_name]
            self.model_type = config["type"]
            
            logger.info(f"Loading {model_name} ({self.model_type})...")
            logger.info(f"  Description: {config['description']}")
            
            if self.model_type == "deepseek":
                return self._load_deepseek_model(model_name, config)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"  ✗ Failed to load OCR model {model_name}: {str(e)}")
            self.model_name = None
            self.model_type = None
            return False
    
    def _load_deepseek_model(self, model_name: str, config: dict) -> bool:
        """Load DeepSeek OCR model."""
        if not DEEPSEEK_AVAILABLE:
            raise RuntimeError("DeepSeek dependencies not available. Install: pip install torch transformers")
        
        model_path = config["path"]
        logger.info(f"  Path: {model_path}")
        
        # Load tokenizer
        logger.info("  Loading OCR tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load model
        logger.info("  Loading OCR model...")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True
        )
        
        # Move to GPU and set to bfloat16
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        logger.info("    ✓ Model loaded successfully")
        
        # Track GPU memory usage
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"  GPU Memory usage across {num_gpus} device(s):")
            total_allocated = 0
            for device_id in range(num_gpus):
                allocated = torch.cuda.memory_allocated(device_id) / 1e9
                total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                device_name = torch.cuda.get_device_name(device_id)
                available = total - allocated
                logger.info(f"    Device {device_id} ({device_name}): {allocated:.2f}GB allocated, {available:.2f}GB available (of {total:.2f}GB total)")
                total_allocated += allocated
            
            total_gpu_memory = sum([torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(num_gpus)])
            logger.info(f"    Total: {total_allocated:.2f}GB allocated across all GPUs (of {total_gpu_memory:.2f}GB total)")
        
        self.model_name = model_name
        logger.info(f"  ✓ Successfully loaded {model_name}")
        return True
    
    def detect_tables_with_ocr(self, image_path: str, output_dir: str = None) -> dict:
        """
        Detect tables in an image using DeepSeek OCR model.
        
        Args:
            image_path: Path to the image file
            output_dir: Optional directory to save OCR results
            
        Returns:
            Dictionary with OCR results including detected tables
        """
        if self.model_name is None:
            raise RuntimeError("OCR model must be loaded first. Call load_model() first.")
        
        logger.info(f"    → Running OCR on: {image_path}")
        
        if self.model_type == "deepseek":
            return self._detect_tables_deepseek(image_path, output_dir)
        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")
    
    def _detect_tables_deepseek(self, image_path: str, output_dir: str = None) -> dict:
        """
        Detect tables using DeepSeek OCR.
        
        Args:
            image_path: Path to the image file
            output_dir: Optional directory to save OCR results
            
        Returns:
            Dictionary with OCR results
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("DeepSeek model not loaded properly.")
        
        # Use grounding prompt to detect tables and get their positions
        prompt ="<image>\n<|grounding|>Convert the document to markdown. "
        
        # Capture stdout since the model prints results instead of returning them
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Run OCR inference
            res = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir if output_dir else '',
                base_size=1280,
                image_size=1280,
                crop_mode=False,
                save_results=True,
                test_compress=False
            )
        finally:
            # Restore stdout
            sys.stdout = original_stdout
        
        # Get the captured output
        output_text = captured_output.getvalue()
        
        logger.info(f"    → OCR complete")
        
        return {
            "markdown": output_text,
            "raw_result": res
        }
    
    def unload_model(self) -> None:
        """Unload the OCR model and free resources."""
        if self.model_name is not None:
            logger.info(f"  Unloading {self.model_name}...")
            
            if self.model_type == "deepseek" and self.model is not None:
                # DeepSeek model cleanup
                if torch and torch.cuda.is_available():
                    allocated_before = torch.cuda.memory_allocated(0) / 1e9
                    logger.info(f"  GPU Memory before unload: {allocated_before:.2f}GB allocated")
                
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                
                if torch:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    if torch.cuda.is_available():
                        allocated_after = torch.cuda.memory_allocated(0) / 1e9
                        total = torch.cuda.get_device_properties(0).total_memory / 1e9
                        freed = allocated_before - allocated_after
                        available = total - allocated_after
                        logger.info(f"  GPU Memory after unload: {allocated_after:.2f}GB allocated, {available:.2f}GB available")
                        logger.info(f"  ✓ Freed {freed:.2f}GB of GPU memory")
            
            self.model_name = None
            self.model_type = None
            logger.info(f"  ✓ Model unloaded")


# Export for use in detect_tables.py
__all__ = ["OCRModelManager", "OCR_MODEL_CONFIG"]
