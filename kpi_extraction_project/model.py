import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from logger import logger

# Base path for shared model weights on the UKP cluster
SHARED_MODELS_BASE = "/storage/ukp/shared/shared_model_weights"

MODEL_CONFIGS = {
    # ...existing configs...
    "deepseek-r1-distill-llama-70b": {
        "path": f"{SHARED_MODELS_BASE}/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
        "includes_prompt_in_output": True,
        "description": "DeepSeek R1 Distill Llama 70B - Distilled reasoning model based on Llama architecture",
        "max_new_tokens": 16384,
        "max_memory": {0: "75GB", 1: "75GB"}
    },
    # ...other configs...
}

class ModelManager:
    def __init__(self, temperature: float = 0.1):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
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
                logger.info(f"  Using multi-GPU setup with memory limits: {max_memory}")
                if llm_int8_enable_fp32_cpu_offload:
                    logger.info(f"  CPU offload enabled for layers that don't fit in GPU")

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
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                available = total - allocated
                logger.info(f"  GPU Memory: {allocated:.2f}GB allocated, {available:.2f}GB available (of {total:.2f}GB total)")

            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_name}: {str(e)}")
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            return False

    def unload_model(self) -> None:
        if self.current_model is not None:
            logger.info(f"  Unloading {self.current_model_name}...")
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"  GPU Memory before unload: {allocated_before:.2f}GB allocated")
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
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

# Export for use in extract_kpis_multi_model.py
__all__ = ["MODEL_CONFIGS", "ModelManager"]