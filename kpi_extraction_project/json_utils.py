#!/usr/bin/env python3
"""
JSON Utility Functions for KPI Extraction
==========================================

Simple utilities to clean and parse JSON responses from LLM models.
Handles common issues like markdown formatting around JSON.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def clean_json_response(text: str, remove_prompt: str = None) -> str:
    """
    Extract JSON from model output by removing text before first { and after last }.
    
    Simple approach: Just find first '{' and last '}' and extract everything between them.
    No complex brace matching, no markdown handling - let max_new_tokens handle completeness.
    
    Args:
        text: Raw model output
        remove_prompt: Original prompt to remove if model echoed it (unused now)
        
    Returns:
        Cleaned JSON string
    """
    # Find the first '{' character (start of JSON object)
    first_brace = text.find('{')
    if first_brace == -1:
        logger.warning(f"    → No JSON object found in response (no '{{' character)")
        return text  # Return as-is for error reporting
    
    # Find the last '}' character (end of JSON object)
    last_brace = text.rfind('}')
    if last_brace == -1:
        logger.warning(f"    → No closing brace found in response (no '}}' character)")
        return text  # Return as-is for error reporting
    
    # Log what we're stripping
    if first_brace > 0:
        logger.info(f"    → Stripping {first_brace} characters before first '{{' character")
    if last_brace < len(text) - 1:
        extra_chars = len(text) - last_brace - 1
        logger.info(f"    → Stripping {extra_chars} characters after last '}}' character")
    
    # Extract everything from first { to last }
    return text[first_brace:last_brace + 1].strip()


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
