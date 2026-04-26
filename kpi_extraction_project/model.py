import os
import torch
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
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
        "max_new_tokens": 16384,
    },
    "Qwen2.5-VL-32B-Instruct": {
        "path": f"{SHARED_MODELS_BASE}/Qwen2.5-VL-32B-Instruct",
        "includes_prompt_in_output": True,
        "description": "Qwen2.5-VL-32B-Instruct - Multimodal model with 32B parameters",
        "max_new_tokens": 16384,
    },
    # ...other configs...
}

class ModelManager:
    def __init__(self, temperature: float = 0.1):
        self.current_llm = None           # vLLM LLM instance (text and VLM)
        self.current_tokenizer = None     # HF tokenizer for chat template formatting
        self.current_model_name = None
        self.current_ocr_model = None
        self.current_ocr_tokenizer = None
        self.temperature = temperature

    def load_model(self, model_name: str) -> bool:
        try:
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]

            logger.info(f"Loading {model_name} with vLLM...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")

            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

            self.current_llm = LLM(
                model=model_path,
                tensor_parallel_size=num_gpus,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                max_model_len=config.get("max_new_tokens", 8192),
                enable_prefix_caching=True,   # reuse KV cache for shared prompt prefixes
                enforce_eager=False,           # use CUDA graphs for faster decoding
                disable_custom_all_reduce=True,  # required when NCCL_P2P_DISABLE=1
            )

            logger.info("  Loading tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            self.current_model_name = model_name
            self._log_gpu_memory()
            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_name}: {str(e)}")
            self.current_llm = None
            self.current_tokenizer = None
            self.current_model_name = None
            return False

    def unload_model(self) -> None:
        has_model = (
            self.current_llm is not None or
            self.current_ocr_model is not None
        )
        if has_model:
            logger.info(f"  Unloading {self.current_model_name}...")
            if self.current_llm is not None:
                del self.current_llm
            if self.current_tokenizer is not None:
                del self.current_tokenizer
            if self.current_ocr_model is not None:
                del self.current_ocr_model
            if self.current_ocr_tokenizer is not None:
                del self.current_ocr_tokenizer
            self.current_llm = None
            self.current_tokenizer = None
            self.current_model_name = None
            self.current_ocr_model = None
            self.current_ocr_tokenizer = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"  ✓ Model unloaded")

    def _log_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"  GPU Memory usage across {num_gpus} device(s):")
            total_allocated = 0
            for device_id in range(num_gpus):
                allocated = torch.cuda.memory_allocated(device_id) / 1e9
                total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                available = total - allocated
                device_name = torch.cuda.get_device_name(device_id)
                logger.info(f"    Device {device_id} ({device_name}): {allocated:.2f}GB allocated, {available:.2f}GB available (of {total:.2f}GB total)")
                total_allocated += allocated
            total_gpu_memory = sum(
                torch.cuda.get_device_properties(i).total_memory / 1e9
                for i in range(num_gpus)
            )
            logger.info(f"    Total: {total_allocated:.2f}GB allocated across all GPUs (of {total_gpu_memory:.2f}GB total)")

    def generate_text(self, prompt: str) -> str:
        """Generate text using vLLM with tokenizer chat template."""
        if self.current_llm is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")

        config = MODEL_CONFIGS[self.current_model_name]
        max_new_tokens = config.get("max_new_tokens", 2048)

        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            top_p=0.95 if self.temperature > 0 else 1.0,
            max_tokens=max_new_tokens,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.current_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.current_llm.generate([text], sampling_params)
        return outputs[0].outputs[0].text

    def load_vlm_model(self, model_name: str) -> bool:
        """Load a Qwen2.5-VL vision-language model via vLLM."""
        try:
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]

            logger.info(f"Loading {model_name} with vLLM (VLM)...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")
            logger.info(f"  Type: VLM (Vision-Language Model)")

            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            logger.info("  Loading VLM model with vLLM...")
            self.current_llm = LLM(
                model=model_path,
                tensor_parallel_size=num_gpus,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                limit_mm_per_prompt={"image": 1},
                max_model_len=config.get("max_new_tokens", 8192),
                enable_prefix_caching=True,   # reuse KV cache for shared prompt prefixes
                enforce_eager=False,           # use CUDA graphs for faster decoding
                disable_custom_all_reduce=True,  # required when NCCL_P2P_DISABLE=1
            )

            logger.info("  Loading tokenizer...")
            self.current_model_name = model_name
            self._log_gpu_memory()
            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to load VLM {model_name}: {str(e)}")
            self.current_llm = None
            self.current_tokenizer = None
            return False

    def generate_vlm_output(self, image_path: str, prompt: str) -> str:
        """Generate output from a VLM given an image path and prompt text."""
        if self.current_llm is None:
            raise RuntimeError("VLM model must be loaded first. Call load_vlm_model() first.")

        config = MODEL_CONFIGS[self.current_model_name]
        max_new_tokens = config.get("max_new_tokens", 2048)

        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            top_p=0.95 if self.temperature > 0 else 1.0,
            max_tokens=max_new_tokens,
        )

        image = Image.open(image_path).convert("RGB")

        # Cap image size to avoid CUDA OOM on very large images.
        # Limit total pixel count rather than max side length so that
        # tall/wide images with manageable area are not needlessly downscaled.
        MAX_PIXELS = 1500 * 1500  # 2,250,000 px
        if image.width * image.height > MAX_PIXELS:
            scale = (MAX_PIXELS / (image.width * image.height)) ** 0.5
            new_size = (int(image.width * scale), int(image.height * scale))
            logger.info(f"    → Resizing image from {image.width}x{image.height} to {new_size[0]}x{new_size[1]} (max pixels: {MAX_PIXELS})")
            image = image.resize(new_size, Image.LANCZOS)

        def _has_image_placeholder(text: str) -> bool:
            markers = (
                "<|image_pad|>",
                "<|vision_start|>",
                "<image>",
                "<img>",
                "<|img|>",
            )
            return any(marker in text for marker in markers)

        prompt_candidates = []
        prompt_build_errors = []
        use_fallbacks = self.current_model_name == "Qwen2.5-VL-7B-Instruct"

        # Normal prompting path (always enabled).
        try:
            text = self.current_tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_candidates.append(("chat_template_mm", text))
        except Exception as e:
            prompt_build_errors.append(f"chat_template_mm: {e}")

        # Fallback prompting paths are only enabled for the 7B model.
        if use_fallbacks:
            try:
                text = self.current_tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": "<|vision_start|><|image_pad|><|vision_end|>\n" + prompt,
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_candidates.append(("chat_template_qwen_vision", text))
            except Exception as e:
                prompt_build_errors.append(f"chat_template_qwen_vision: {e}")

            prompt_candidates.append(
                (
                    "raw_qwen_vision",
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                    + prompt
                    + "\n<|im_end|>\n<|im_start|>assistant\n",
                )
            )

        if not prompt_candidates:
            raise RuntimeError(
                "Failed to format VLM prompt for model "
                f"'{self.current_model_name}'. Errors: {'; '.join(prompt_build_errors)}"
            )

        logger.info(f"    → Generating with vLLM...")
        last_generate_error = None
        outputs = None

        for attempt in range(2):
            outputs = None

            for prompt_name, text in prompt_candidates:
                if not _has_image_placeholder(text):
                    logger.info(f"    → Skipping prompt variant without image marker: {prompt_name}")
                    continue

                if use_fallbacks:
                    generation_inputs = [
                        {
                            "prompt": text,
                            "multi_modal_data": {"image": image},
                        },
                        {
                            "prompt": text,
                            "multi_modal_data": {"image": [image]},
                        },
                        [
                            {
                                "prompt": text,
                                "multi_modal_data": {"image": image},
                            }
                        ],
                        [
                            {
                                "prompt": text,
                                "multi_modal_data": {"image": [image]},
                            }
                        ],
                    ]
                else:
                    generation_inputs = [
                        {
                            "prompt": text,
                            "multi_modal_data": {"image": image},
                        }
                    ]

                for input_variant in generation_inputs:
                    try:
                        outputs = self.current_llm.generate(input_variant, sampling_params)
                        logger.info(f"    → Using prompt variant: {prompt_name}")
                        break
                    except Exception as e:
                        last_generate_error = e

                if outputs is not None:
                    break

            if outputs is not None:
                break

            if attempt == 0 and last_generate_error is not None and self._is_engine_core_error(last_generate_error):
                model_name = self.current_model_name
                logger.warning(f"    → EngineCore failure detected, restarting vLLM engine and retrying...")
                self._restart_vlm_engine(model_name)
            else:
                break

        if outputs is None:
            raise RuntimeError(
                f"Failed VLM generation for model '{self.current_model_name}': {last_generate_error}"
            )

        output_text = outputs[0].outputs[0].text
        logger.info(f"    → Generation complete.")
        return output_text


    def _is_engine_core_error(self, error: Exception) -> bool:
        error_str = str(error)
        return "EngineCore" in error_str or "Worker failed" in error_str

    def _restart_vlm_engine(self, model_name: str) -> None:
        logger.warning(f"    → Destroying broken vLLM engine...")
        if self.current_llm is not None:
            del self.current_llm
            self.current_llm = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning(f"    → Reloading {model_name}...")
        if not self.load_vlm_model(model_name):
            raise RuntimeError(f"Failed to restart vLLM engine for model '{model_name}'")
        logger.warning(f"    → Engine restarted successfully.")


# Export for use in extract_kpis_multi_model.py
__all__ = ["MODEL_CONFIGS", "ModelManager"]