"""
Gemini AI Integration for yan.py - FIXED VERSION
Improved response handling to prevent mid-sentence truncation
"""

import os
import requests
import json
from typing import Optional, Dict, Any, List
import logging
import re


class GeminiLLM:
    """
    Wrapper for Google Gemini API with web search support
    Compatible with llama_cpp interface for drop-in replacement
    
    FIXES:
    - Better sentence completion detection
    - Handles all finish reasons (not just MAX_TOKENS)
    - Increased default max_tokens
    - Better error messages
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        enable_web_search: bool = True,
    ):
        """
        Initialize Gemini LLM

        Args:
            api_key: Google API key (or set GEMINI_API_KEY env var)
            model: Model name (gemini-2.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp-0205)
            enable_web_search: Enable Google Search grounding
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.model = model
        self.enable_web_search = enable_web_search
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        logging.info(f"✅ Gemini LLM initialized with model: {model}")
        if enable_web_search:
            logging.info("🌐 Web search enabled via Google Search tool")

    def _is_sentence_complete(self, text: str) -> bool:
        """
        Check if the text ends with a complete sentence
        
        IMPROVED: Better detection of incomplete sentences

        Args:
            text: Text to check

        Returns:
            True if text appears to end with a complete sentence
        """
        if not text:
            return False

        # Strip trailing whitespace
        text = text.rstrip()
        
        if not text:
            return False

        # Check if ends with sentence-ending punctuation
        if text[-1] in ".!?\"')]}":
            return True
        
        # Check for common complete endings
        complete_endings = [
            "etc.",
            "Dr.",
            "Mr.",
            "Mrs.",
            "Ms.",
            "Prof.",
            "vs.",
            "i.e.",
            "e.g.",
        ]
        
        for ending in complete_endings:
            if text.endswith(ending):
                return True

        # FIX 1: Check for incomplete sentence patterns
        incomplete_patterns = [
            r'\w+,$',  # Ends with comma
            r'\w+ (and|or|but|because|if|when|while|since|although)$',  # Ends with conjunction
            r'\w+ (the|a|an|this|that|these|those)$',  # Ends with article
            r'\w+ (is|are|was|were|will|would|should|can|could)$',  # Ends with verb
            r'\w+\s*$',  # Ends with word and whitespace (likely cut off)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        # Default: if doesn't match incomplete patterns, consider complete
        return True

    def _detect_current_topics(self, prompt: str) -> bool:
        """
        Detect if the query is about current/recent information

        Args:
            prompt: User's query

        Returns:
            True if query appears to need current information
        """
        current_indicators = [
            "current",
            "latest",
            "recent",
            "today",
            "now",
            "this year",
            "what is happening",
            "news",
            "update",
            "2024",
            "2025",
            "2026",
            "right now",
            "as of",
            "currently",
        ]

        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in current_indicators)

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 8192,  # FIX 2: Set to Gemini's actual limit
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list = None,
        force_web_search: bool = None,
        _is_retry: bool = False,  # FIX 3: Prevent infinite recursion
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate response from Gemini (llama_cpp compatible interface)

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default 8192 - Gemini's limit)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            force_web_search: Override automatic web search detection
            _is_retry: Internal flag to prevent infinite recursion

        Returns:
            Dict with 'choices' list containing 'text' field
        """
        try:
            # Determine if web search should be used
            use_search = force_web_search
            if use_search is None:
                use_search = self.enable_web_search and self._detect_current_topics(
                    prompt
                )

            # Prepare request
            headers = {"Content-Type": "application/json"}

            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "maxOutputTokens": min(max_tokens, 8192),  # FIX 4: Enforce Gemini limit
                    "candidateCount": 1,
                },
            }

            # Use google_search tool for grounding
            if use_search:
                data["tools"] = [{"google_search": {}}]
                logging.info("🔍 Using Google Search grounding for current information")

            # Add stop sequences if provided
            if stop:
                data["generationConfig"]["stopSequences"] = stop

            # Make API request
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=90)  # FIX 5: Longer timeout

            if response.status_code != 200:
                error_msg = (
                    f"Gemini API error: {response.status_code} - {response.text}"
                )
                logging.error(error_msg)
                raise Exception(error_msg)

            result = response.json()

            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]

                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0]["text"]
                    finish_reason = candidate.get("finishReason", "STOP")

                    # FIX 6: Handle ALL truncation cases, not just MAX_TOKENS
                    is_truncated = (
                        finish_reason in ["MAX_TOKENS", "LENGTH"] or 
                        not self._is_sentence_complete(text)
                    )
                    
                    if is_truncated and not _is_retry:
                        logging.warning(f"⚠️ Response may be truncated (reason: {finish_reason})")
                        logging.warning(f"   Last 100 chars: ...{text[-100:]}")
                        
                        # FIX 7: Try continuation ONLY if not already a retry
                        if not self._is_sentence_complete(text):
                            logging.info("🔄 Attempting to complete truncated response...")

                            # Create continuation prompt
                            continuation_prompt = f"""Continue this response EXACTLY where it left off. 
Complete the sentence and finish the full answer.

Original question: {prompt}

Incomplete response:
{text}

Continue from where it stopped (start with the next word):"""

                            try:
                                # Make ONE retry attempt
                                completion_response = self.__call__(
                                    continuation_prompt,
                                    max_tokens=min(4096, max_tokens // 2),  # Use less tokens for completion
                                    temperature=temperature,
                                    top_p=top_p,
                                    force_web_search=False,  # Don't search again
                                    _is_retry=True,  # Prevent further recursion
                                )

                                completion_text = completion_response["choices"][0]["text"].strip()
                                
                                # FIX 8: Smart merging - avoid duplication
                                if completion_text:
                                    # Remove any repeated content from the start of completion
                                    last_words = " ".join(text.split()[-10:])
                                    if completion_text.startswith(last_words):
                                        # Completion repeats context, skip it
                                        completion_text = completion_text[len(last_words):].lstrip()
                                    
                                    text = text.rstrip() + " " + completion_text
                                    logging.info("✅ Response completion successful")
                                    logging.info(f"   Final length: {len(text)} chars")

                            except Exception as e:
                                logging.error(f"❌ Failed to complete response: {e}")
                                logging.warning("   Returning incomplete response")

                    # Extract grounding metadata if present
                    grounding_metadata = candidate.get("groundingMetadata", {})
                    search_queries = grounding_metadata.get("searchEntryPoint", {}).get(
                        "renderedContent", ""
                    )
                    web_search_queries = grounding_metadata.get("webSearchQueries", [])
                    grounding_supports = grounding_metadata.get("groundingSupports", [])

                    # Return in llama_cpp format
                    response_data = {
                        "choices": [{"text": text, "finish_reason": finish_reason}],
                        "usage": result.get("usageMetadata", {}),
                        "model": self.model,
                    }

                    # Add grounding info if web search was used
                    if use_search and (
                        search_queries or web_search_queries or grounding_supports
                    ):
                        response_data["grounding"] = {
                            "search_queries": web_search_queries,
                            "search_entry_point": search_queries,
                            "supports": grounding_supports,
                        }
                        logging.info(
                            f"📚 Used web search with {len(grounding_supports)} sources"
                        )

                    return response_data
                else:
                    # FIX 9: Handle safety blocks
                    if "safetyRatings" in candidate:
                        safety_ratings = candidate["safetyRatings"]
                        blocked_reasons = [
                            r for r in safety_ratings 
                            if r.get("blocked", False)
                        ]
                        if blocked_reasons:
                            raise Exception(
                                f"Response blocked by safety filters: {blocked_reasons}"
                            )
                    
                    raise Exception(f"No content in response. Candidate: {candidate}")

            raise Exception(f"Unexpected response format from Gemini API: {result.keys()}")

        except requests.exceptions.Timeout:
            logging.error("Gemini API request timed out")
            raise Exception(
                "Gemini API timeout - try reducing max_tokens or simplifying query"
            )
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise


def create_gemini_llm(
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    enable_web_search: bool = True,
) -> GeminiLLM:
    """
    Factory function to create Gemini LLM instance

    Args:
        api_key: Google API key
        model: Gemini model name
               - gemini-2.5-flash (recommended, fast & stable)
               - gemini-1.5-pro (more capable, slower)
               - gemini-2.0-flash-exp (experimental, may not always be available)
        enable_web_search: Enable Google Search grounding for current info

    Returns:
        GeminiLLM instance
    """
    return GeminiLLM(api_key=api_key, model=model, enable_web_search=enable_web_search)
