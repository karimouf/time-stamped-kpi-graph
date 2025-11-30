#!/usr/bin/env python3
"""
JSON Utility Functions for KPI Extraction
==========================================

Simple utilities to clean and parse JSON responses from LLM models.
Handles common issues like markdown formatting around JSON.
"""

import json
import logging

logger = logging.getLogger(__name__)


def clean_json_response(text: str, remove_prompt: str = None) -> str:
    """
    Extract JSON from model output, skipping any <think>...</think> reasoning blocks.
    
    DeepSeek-R1 models output chain-of-thought reasoning in <think> tags before JSON.
    There can be multiple <think> blocks. We need to find the LAST </think> tag and
    extract JSON from the text that comes after it.
    
    Args:
        text: Raw model output (may contain <think> tags and JSON)
        remove_prompt: Original prompt to remove if model echoed it (unused now)
        
    Returns:
        Cleaned JSON string
    """
    # Find the LAST </think> tag - this marks the end of reasoning
    last_think_end = text.rfind('</think>')
    
    if last_think_end != -1:
        # Start searching for JSON AFTER the last </think> tag
        search_start = last_think_end + len('</think>')
        logger.info(f"    → Found </think> tag at position {last_think_end}, searching for JSON after that")
    else:
        # No <think> tags found, search from beginning
        search_start = 0
        logger.info(f"    → No </think> tags found, searching entire output for JSON")
    
    # Extract the text after the last </think> (or all text if no tags)
    json_text = text[search_start:]
    
    # Find the first '{' character (start of JSON object)
    first_brace = json_text.find('{')
    if first_brace == -1:
        logger.warning(f"    → No JSON object found in response (no '{{' character)")
        return text  # Return as-is for error reporting
    
    # Find the last '}' character (end of JSON object)
    last_brace = json_text.rfind('}')
    if last_brace == -1:
        logger.warning(f"    → No closing brace found in response (no '}}' character)")
        return text  # Return as-is for error reporting
    
    # Log what we're stripping
    if first_brace > 0:
        logger.info(f"    → Stripping {first_brace} characters before first '{{' character")
    if last_brace < len(json_text) - 1:
        extra_chars = len(json_text) - last_brace - 1
        logger.info(f"    → Stripping {extra_chars} characters after last '}}' character")
    
    # Extract everything from first { to last } in the post-think text
    return json_text[first_brace:last_brace + 1].strip()


def parse_json_safely(text: str) -> dict:
    """
    Attempt to parse JSON with error handling.
    
    Args:
        text: JSON string to parse
        
    Returns:
        Parsed dictionary or error dictionary
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"    JSON parsing failed: {str(e)}")
        logger.warning(f"    Error at position {e.pos}")
        return None
