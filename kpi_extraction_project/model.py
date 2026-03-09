import os
# os.environ["TRANSFORMERS_VERBOSITY"] = "info"

# Configure logging for the module
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from logger import logger

# Base path for shared model weights on the UKP cluster
SHARED_MODELS_BASE = "/storage/ukp/shared/shared_model_weights"

MODEL_CONFIGS = {
    # ...existing configs...
    # "deepseek-r1-distill-llama-70b": {
    #     "path": f"{SHARED_MODELS_BASE}/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
    #     "includes_prompt_in_output": True,
    #     "description": "DeepSeek R1 Distill Llama 70B - Distilled reasoning model based on Llama architecture",
    #     "max_new_tokens": 16384,
    # },
    "deepseek-ocr": {
        "path": f"{SHARED_MODELS_BASE}/models--deepseek-ai--DeepSeek-OCR",
        "includes_prompt_in_output": False,
        "description": "DeepSeek OCR - Vision-language model for OCR and table detection with grounding",
        "max_new_tokens": 8192,
        "model_type": "ocr",
    },
    "Qwen2.5-VL-7B-Instruct": {
        "path": f"{SHARED_MODELS_BASE}/Qwen2.5-VL-7B-Instruct",
        "includes_prompt_in_output": True,
        "description": "Qwen2.5-VL-7B-Instruct - Multimodal model with 7B parameters",
        "max_new_tokens": 8192,
    },
    "Qwen2.5-VL-72B-Instruct": {
        "path": f"{SHARED_MODELS_BASE}/Qwen2.5-VL-72B-Instruct",
        "includes_prompt_in_output": True,
        "description": "Qwen2.5-VL-72B-Instruct - Multimodal model with 72B parameters",
        "max_new_tokens": 8192,
    },
    "Qwen2.5-VL-32B-Instruct": {
        "path": f"{SHARED_MODELS_BASE}/Qwen2.5-VL-32B-Instruct",
        "includes_prompt_in_output": True,
        "description": "Qwen2.5-VL-32B-Instruct - Multimodal model with 32B parameters",
        "max_new_tokens": 8192,
    },
    # ...other configs...
}

class ModelManager:
    def __init__(self, temperature: float = 0.1):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.current_vlm_model = None
        self.current_vlm_processor = None
        self.current_ocr_model = None
        self.current_ocr_tokenizer = None
        self.temperature = temperature

    def load_model(self, model_name: str) -> bool:
        try:
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]

            logger.info(f"Loading {model_name}...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")

            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                legacy=False,
                padding_side="left",
                trust_remote_code=True
            )

            quantization_config = None
            llm_int8_enable_fp32_cpu_offload = config.get("llm_int8_enable_fp32_cpu_offload", False)

            if config.get("quantization") == "4bit":
                logger.info(f"  Using 4-bit NF4 quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
                )
            elif config.get("quantization") == "8bit":
                logger.info(f"  Using 8-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
                )

            max_memory = config.get("max_memory", None)
            if max_memory:
                logger.info(f"  Using configured memory limits: {max_memory}")
            else:
                logger.info(f"  Using automatic memory allocation (device_map='auto')")

            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16 if quantization_config is None else None,
                quantization_config=quantization_config,
                trust_remote_code=True
            )

            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.current_model.resize_token_embeddings(len(self.current_tokenizer))
                self.current_model.config.pad_token_id = self.current_tokenizer.pad_token_id
                self.current_model.generation_config.pad_token_id = self.current_tokenizer.pad_token_id

            self.current_model_name = model_name

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                logger.info(f"  GPU Memory usage across {num_gpus} device(s):")
                total_allocated = 0
                total_available = 0
                for device_id in range(num_gpus):
                    allocated = torch.cuda.memory_allocated(device_id) / 1e9
                    reserved = torch.cuda.memory_reserved(device_id) / 1e9
                    total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                    available = total - allocated
                    device_name = torch.cuda.get_device_name(device_id)
                    logger.info(f"    Device {device_id} ({device_name}): {allocated:.2f}GB allocated, {available:.2f}GB available (of {total:.2f}GB total)")
                    total_allocated += allocated
                    total_available += sum([torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(num_gpus)]) - total_allocated
                
                total_gpu_memory = sum([torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(num_gpus)])
                logger.info(f"    Total: {total_allocated:.2f}GB allocated across all GPUs (of {total_gpu_memory:.2f}GB total)")

            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_name}: {str(e)}")
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            return False

    def unload_model(self) -> None:
        has_model = (self.current_model is not None or 
                     self.current_vlm_model is not None or 
                     self.current_ocr_model is not None)
        if has_model:
            logger.info(f"  Unloading {self.current_model_name}...")
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"  GPU Memory before unload: {allocated_before:.2f}GB allocated")
            if self.current_model is not None:
                del self.current_model
            if self.current_tokenizer is not None:
                del self.current_tokenizer
            if self.current_vlm_model is not None:
                del self.current_vlm_model
            if self.current_vlm_processor is not None:
                del self.current_vlm_processor
            if self.current_ocr_model is not None:
                del self.current_ocr_model
            if self.current_ocr_tokenizer is not None:
                del self.current_ocr_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            self.current_vlm_model = None
            self.current_vlm_processor = None
            self.current_ocr_model = None
            self.current_ocr_tokenizer = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                freed = allocated_before - allocated_after
                available = total - allocated_after
                logger.info(f"  GPU Memory after unload: {allocated_after:.2f}GB allocated, {available:.2f}GB available")
                logger.info(f"  ✓ Freed {freed:.2f}GB of GPU memory")
            else:
                logger.info(f"  ✓ Model unloaded")
    
    def generate_text(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text using the current model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Generated text (decoded, without input prompt)
        """
        if self.current_model is None or self.current_tokenizer is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.current_model.device)
        
        input_length = inputs['input_ids'].shape[1]
        
        # Get model-specific max_new_tokens limit
        config = MODEL_CONFIGS[self.current_model_name]
        max_new_tokens = config.get("max_new_tokens", 2048)
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.current_tokenizer.pad_token_id,
            "eos_token_id": self.current_tokenizer.eos_token_id
        }
        
        # Only add sampling parameters if sampling is enabled
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95
        
        # Generate response
        with torch.inference_mode():
            outputs = self.current_model.generate(**inputs, **gen_kwargs)
        
        # Decode only the newly generated tokens (skip input prompt)
        generated_ids = outputs[0][input_length:]
        generated_text = self.current_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_text
    
    def load_vlm_model(self, model_name: str) -> bool:
        """
        Load a vision-language model (VLM) with GPU memory tracking.
        
        Args:
            model_name: Name of the VLM model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]
            logger.info(f"Loading {model_name}...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")
            logger.info(f"  Type: VLM (Vision-Language Model)")

            
            # Load processor with recommended pixel range (256-1280 tokens)
            logger.info("  Loading VLM processor...")
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            self.current_vlm_processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            logger
            logger.info("  Loading VLM model...")
            if "7b" in model_name or "32b" in model_name:
                device_map = "cuda:0"  # Load smaller models on a single GPU
            else:
                device_map = "auto"  # Let larger models be distributed across multiple GPUs

            # Load with optimizations to reduce inter-GPU transfer overhead
            self.current_vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,  # Reduce CPU memory usage during load
            )
            
            # Track GPU memory usage
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                logger.info(f"  GPU Memory usage across {num_gpus} device(s):")
                total_allocated = 0
                for device_id in range(num_gpus):
                    allocated = torch.cuda.memory_allocated(device_id) / 1e9
                    reserved = torch.cuda.memory_reserved(device_id) / 1e9
                    total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                    available = total - allocated
                    device_name = torch.cuda.get_device_name(device_id)
                    logger.info(f"    Device {device_id} ({device_name}): {allocated:.2f}GB allocated, {available:.2f}GB available (of {total:.2f}GB total)")
                    total_allocated += allocated
                
                total_gpu_memory = sum([torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(num_gpus)])
                logger.info(f"    Total: {total_allocated:.2f}GB allocated across all GPUs (of {total_gpu_memory:.2f}GB total)")
            
            self.current_model_name = model_name
            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load VLM {model_name}: {str(e)}")
            self.current_vlm_model = None
            self.current_vlm_processor = None
            return False
        
        
    def _get_input_device(self):
        """Get the device where input tensors should be placed."""
        if hasattr(self.current_vlm_model, 'hf_device_map'):
            # Find the device of the first layer/embedding
            device_map = self.current_vlm_model.hf_device_map
            # The visual encoder or embedding layer handles inputs first
            for key in ['visual', 'model.embed_tokens', 'model', '']:
                if key in device_map:
                    return f"cuda:{device_map[key]}" if isinstance(device_map[key], int) else device_map[key]
        # Fallback
        return next(self.current_vlm_model.parameters()).device
    
    def generate_vlm_output(self, image_path: str, prompt: str) -> str:
        """
        Generate output from a vision-language model (VLM) given an image and prompt.
        
        Args:
            image_path: Path to the image file or URL
            prompt: Text prompt to send to the model
            
        Returns:
            Generated text output from the model
        """
        if self.current_vlm_model is None or self.current_vlm_processor is None:
            raise RuntimeError("VLM model and processor must be loaded first. Call load_vlm_model() first.")
        
        # Get model-specific max_new_tokens limit
        config = MODEL_CONFIGS[self.current_model_name]
        max_new_tokens = config.get("max_new_tokens", 2048)
        
        # Prepare messages in the format expected by Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Preparation for inference (using official Qwen2.5-VL pattern)
        logger.info(f"    → Processing image and prompt...")
        
        # Apply chat template to format the prompt
        text = self.current_vlm_processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision information (images/videos)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs with processor
        inputs = self.current_vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to CUDA
        input_device = self._get_input_device()
        inputs = inputs.to(input_device)
        
            # Prepare generation kwargs
        if self.temperature > 0:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": 0.95,
            }   
        else:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": None,  # explicitly null to avoid warnings
                "top_p": None,
            }
        
        logger.info(f"    → Generating tokens with gen_kwargs: {gen_kwargs}")
        
        # Inference: Generation of the output
        with torch.inference_mode():
            generated_ids = self.current_vlm_model.generate(**inputs, **gen_kwargs)
        
        logger.info(f"    → Generation complete, decoding output...")

        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        logger.info(f"    → Decoding generated tokens...")
        # Decode the output
        output_text = self.current_vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"    → Generation complete.")
        
        return output_text
    
    def load_ocr_model(self, model_name: str = "deepseek-ocr") -> bool:
        """
        Load DeepSeek OCR model for table detection and OCR tasks.
        
        Args:
            model_name: Name of the OCR model to load (default: "deepseek-ocr")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]

            logger.info(f"Loading {model_name}...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")
            logger.info(f"  Type: OCR Model")
            
            # Load tokenizer
            logger.info("  Loading OCR tokenizer...")
            self.current_ocr_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load model (flash_attention_2 removed from args - let model auto-detect)
            logger.info("  Loading OCR model...")
            self.current_ocr_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True
            )
            
            # Move to GPU and set to bfloat16
            self.current_ocr_model = self.current_ocr_model.eval().cuda().to(torch.bfloat16)
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
            
            self.current_model_name = model_name
            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load OCR model {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.current_ocr_model = None
            self.current_ocr_tokenizer = None
            return False
    
    def detect_tables_with_ocr(self, image_path: str, output_dir: str = None) -> dict:
        """
        Detect tables in an image using DeepSeek OCR with grounding.
        
        Args:
            image_path: Path to the image file
            output_dir: Optional directory to save OCR results
            
        Returns:
            Dictionary with OCR results including detected tables with bboxes and titles
        """
        if self.current_ocr_model is None or self.current_ocr_tokenizer is None:
            raise RuntimeError("OCR model must be loaded first. Call load_ocr_model() first.")
        
        # Use grounding prompt to detect tables and get their positions
        prompt = """Extract all tables with titles, return JSON with:
- table_title
- headers
- rows
- page & bounding boxes """
        
        logger.info(f"    → Running OCR on: {image_path}")
        
        # Run OCR inference
        # Parameters matching "Gundam" configuration from DeepSeek OCR README
        res = self.current_ocr_model.infer(
            self.current_ocr_tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir if output_dir else '',
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True
        )
        
        logger.info(f"    → OCR complete")
        
        return res

# Export for use in extract_kpis_multi_model.py
__all__ = ["MODEL_CONFIGS", "ModelManager"]