"""
CustomLLM - Robust LLM interface for Dynamic-KG

Provides a flexible interface for interacting with large language models,
supporting both standard completions and structured output extraction.
"""

import json
import time
import asyncio
import logging
import re
import tiktoken
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

# OpenAI imports
try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available, OpenAI models will not function")

# Type aliases for better readability
UsageDict = Dict[str, Optional[int]]
JsonSchema = Dict[str, Any]

class CustomLLM:
    """
    Flexible LLM interface with robust error handling and structured output support.
    
    Features:
    - Async and sync request handling
    - Structured output via function/tool calling
    - Fallback extraction from standard completions
    - Token usage tracking
    - Retry logic with backoff
    """

    DEFAULT_TEMP = 0.5
    DEFAULT_MAX_TOKENS = 2000
    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 3
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        default_temperature: float = DEFAULT_TEMP,
        logging_level: int = logging.INFO
    ):
        """
        Initialize the LLM interface.
        
        Args:
            api_key: API key for the model provider (e.g., OpenAI)
            model: Model identifier to use
            default_temperature: Default sampling temperature
            logging_level: Logging level for this class
        """
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        
        # Configure logger
        self.logger = logging.getLogger("core.models.customllm")
        self.logger.setLevel(logging_level)
        
        # Initialize clients
        self._sync_client = None
        self._async_client = None
        
        # Load tokenizer for token counting
        self.tokenizer = self._get_tokenizer(model)
        
        if api_key and OPENAI_AVAILABLE:
            self._init_clients()
            
        self.logger.info(f"CustomLLM initialized with model: {model}")
    
    def _init_clients(self):
        """Initialize OpenAI sync and async clients."""
        try:
            self._sync_client = OpenAI(api_key=self.api_key)
            self.logger.debug("OpenAI synchronous client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI sync client: {str(e)}")
            self._sync_client = None
    
    def _get_async_client(self) -> AsyncOpenAI:
        """Lazy initialization of async client."""
        if self._async_client is None:
            try:
                self._async_client = AsyncOpenAI(api_key=self.api_key)
                self.logger.info("AsyncOpenAI client initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize AsyncOpenAI client: {str(e)}")
                raise RuntimeError(f"Could not initialize AsyncOpenAI client: {str(e)}") from e
        return self._async_client
    
    def _get_tokenizer(self, model: str):
        """Get appropriate tokenizer for token counting."""
        try:
            if "gpt-4" in model:
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base for newer models
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer: {e}. Will use approximate counting.")
            return None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using appropriate tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate token count (4 chars per token)
            return len(text) // 4
    
    def _parse_usage(self, response: Any) -> UsageDict:
        """
        Extract token usage info from response.
        
        Args:
            response: Response from LLM API
            
        Returns:
            Dictionary with token usage counts
        """
        usage_data = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        
        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                usage_data["prompt_tokens"] = getattr(usage, 'prompt_tokens', None)
                usage_data["completion_tokens"] = getattr(usage, 'completion_tokens', None)
                usage_data["total_tokens"] = getattr(usage, 'total_tokens', None)
                self.logger.debug(f"Parsed token usage: {usage_data}")
            else:
                self.logger.debug("No usage information found in LLM response")
        except Exception as e:
            self.logger.warning(f"Error parsing token usage: {e}")
            
        return usage_data
    
    async def _make_llm_call_with_retry(
        self, 
        is_async: bool = True, 
        **kwargs
    ) -> Any:
        """
        Make API call with retry logic.
        
        Args:
            is_async: Whether to use async client
            **kwargs: Arguments for API call
            
        Returns:
            API response
            
        Raises:
            Exception: If all retries fail
        """
        max_retries = self.MAX_RETRIES
        retry_delay = self.RETRY_DELAY_SECONDS
        
        for attempt in range(max_retries + 1):
            try:
                if is_async:
                    client = self._get_async_client()
                    response = await client.chat.completions.create(**kwargs)
                else:
                    if not self._sync_client:
                        raise RuntimeError("Synchronous client not initialized")
                    response = self._sync_client.chat.completions.create(**kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Retry successful (attempt {attempt + 1})")
                    
                return response
                
            except Exception as e:
                # Check for rate limit error
                is_rate_limit = "rate limit" in str(e).lower()
                
                if is_rate_limit and attempt < max_retries:
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}). "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    
                    if is_async:
                        await asyncio.sleep(retry_delay)
                    else:
                        time.sleep(retry_delay)
                        
                    # Increase delay for next attempt
                    retry_delay *= 2
                else:
                    self.logger.error(
                        f"LLM call failed (attempt {attempt + 1}): {str(e)}",
                        exc_info=(not is_rate_limit)
                    )
                    raise
        
        # Should never reach here if exceptions are raised correctly
        raise RuntimeError("LLM call failed after maximum retries")

    #
    # Text Completion Methods
    #
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None
    ) -> Tuple[str, UsageDict]:
        """
        Generate text completion (async).
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            
        Returns:
            Tuple of (generated text, token usage dict)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature if temperature is not None else self.default_temperature
        
        self.logger.debug(
            f"Requesting completion from {self.model} "
            f"(max_tokens={max_tokens}, temp={temp})"
        )
        
        response = await self._make_llm_call_with_retry(
            is_async=True,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp
        )
        
        result_str = ""
        if response.choices:
            result_str = response.choices[0].message.content
            result_str = result_str.strip() if result_str else ""
        else:
            self.logger.warning("LLM response did not contain any choices")
        
        usage_dict = self._parse_usage(response)
        return (result_str, usage_dict)
    
    def generate_completion_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None
    ) -> Tuple[str, UsageDict]:
        """
        Generate text completion (sync).
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            
        Returns:
            Tuple of (generated text, token usage dict)
        """
        if not self._sync_client:
            self.logger.error("Synchronous client not initialized")
            return ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature if temperature is not None else self.default_temperature
        
        try:
            response = self._make_llm_call_with_retry(
                is_async=False,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp
            )
            
            result_str = ""
            if response.choices:
                result_str = response.choices[0].message.content
                result_str = result_str.strip() if result_str else ""
            
            usage_dict = self._parse_usage(response)
            return (result_str, usage_dict)
            
        except Exception as e:
            self.logger.error(f"Sync completion failed: {str(e)}")
            return ("", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})
    
    #
    # Structured Output Methods
    #
    
    def _sanitize_schema(self, schema: JsonSchema) -> JsonSchema:
        """
        Ensure schema is valid for API function calling.
        
        Args:
            schema: JSON schema to sanitize
            
        Returns:
            Sanitized schema
        """
        sanitized = json.loads(json.dumps(schema))
        
        # Ensure top level is object type
        if "type" not in sanitized:
            sanitized["type"] = "object"
            
        if sanitized.get("type") == "object" and "properties" not in sanitized:
            sanitized["properties"] = {}
            
        # Handle arrays
        if "items" in sanitized and isinstance(sanitized.get("items"), dict):
            if "type" not in sanitized["items"]:
                sanitized["items"]["type"] = "object"
                
        # Process nested properties
        self._sanitize_schema_properties(sanitized)
        
        return sanitized
    
    def _sanitize_schema_properties(self, schema_obj: JsonSchema) -> None:
        """
        Recursively sanitize schema properties.
        
        Args:
            schema_obj: Schema object to sanitize
        """
        if "properties" in schema_obj and isinstance(schema_obj["properties"], dict):
            for prop_name, prop_schema in schema_obj["properties"].items():
                if isinstance(prop_schema, dict):
                    if "type" not in prop_schema:
                        prop_schema["type"] = "string"
                        
                    if prop_schema.get("type") == "object":
                        self._sanitize_schema_properties(prop_schema)
                        
                    if prop_schema.get("type") == "array" and "items" in prop_schema:
                        if isinstance(prop_schema["items"], dict):
                            if "type" not in prop_schema["items"]:
                                prop_schema["items"]["type"] = "object"
                            self._sanitize_schema_properties(prop_schema["items"])
        
        # Handle array items
        if "items" in schema_obj and isinstance(schema_obj["items"], dict):
            if "type" not in schema_obj["items"]:
                schema_obj["items"]["type"] = "object"
            self._sanitize_schema_properties(schema_obj["items"])
    
    async def generate_structured_output(
        self,
        prompt: str,
        output_schema: JsonSchema,
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        description: str = "structured data"
    ) -> Tuple[Dict[str, Any], UsageDict]:
        """
        Generate structured output using function/tool calling.
        
        Args:
            prompt: User prompt text
            output_schema: JSON schema for desired output structure
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            description: Description of the output for error messages
            
        Returns:
            Tuple of (structured output dict, token usage dict)
            
        Raises:
            ValueError: If structured output generation fails
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        tool_name = "extract_structured_data"
        sanitized_schema = self._sanitize_schema(output_schema)
        temp = temperature if temperature is not None else self.default_temperature
        
        self.logger.debug(
            f"Requesting structured output from {self.model} "
            f"(max_tokens={max_tokens}, temp={temp})"
        )
        
        try:
            response = await self._make_llm_call_with_retry(
                is_async=True,
                model=self.model,
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Extract {description} based on the provided schema.",
                        "parameters": sanitized_schema
                    }
                }],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                max_tokens=max_tokens,
                temperature=temp
            )
            
            usage_dict = self._parse_usage(response)
            result_dict = {}
            
            message = response.choices[0].message if response.choices else None
            if message and message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == tool_name:
                    try:
                        arguments = tool_call.function.arguments
                        result_dict = json.loads(arguments)
                        self.logger.info(f"Successfully parsed structured output")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON from function call: {e}")
                        raise ValueError(f"Invalid JSON in function arguments: {e}") from e
                else:
                    self.logger.error(f"Unexpected function called: {tool_call.function.name}")
                    raise ValueError(f"Unexpected function called: {tool_call.function.name}")
            else:
                # Fallback to parsing from content
                self.logger.warning("LLM did not use function call, attempting to extract JSON from content")
                content = message.content.strip() if message and message.content else ""
                
                if content:
                    parsed_fallback = self._extract_and_parse_json(content)
                    if parsed_fallback is not None:
                        result_dict = parsed_fallback
                        self.logger.info("Parsed content fallback as JSON")
                    else:
                        self.logger.error(f"Content could not be parsed as JSON")
                        raise ValueError("LLM did not use function call and content was not valid JSON")
                else:
                    self.logger.error("LLM returned empty content")
                    raise ValueError("LLM did not use function call and returned empty content")
            
            return (result_dict, usage_dict)
            
        except Exception as e:
            if "Invalid schema" in str(e) and "schema must be a JSON Schema" in str(e):
                self.logger.error(
                    f"Schema validation error. Schema: {json.dumps(sanitized_schema, indent=2)}"
                )
            raise
    
    def _extract_and_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text (handles markdown blocks).
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON or None if parsing fails
        """
        # Try to find JSON in markdown code blocks
        match = re.search(r"```(?:json)?\s*({.*?}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            # Try to find JSON object/array directly
            start_brace = text.find('{')
            start_bracket = text.find('[')
            
            if start_brace == -1 and start_bracket == -1:
                return None
                
            if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                # Find matching closing brace (simplified approach)
                json_str = text[start_brace:]
            elif start_bracket != -1:
                # Find matching closing bracket (simplified approach)
                json_str = text[start_bracket:]
            else:
                return None
        
        if not json_str:
            return None
            
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            self.logger.debug(f"Failed to parse extracted JSON: {json_str[:200]}...")
            return None
    
    async def generate_and_parse_json(
        self,
        prompt: str,
        description: str = "the required data",
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], UsageDict]:
        """
        Generate text completion and parse JSON from response.
        
        Args:
            prompt: User prompt text
            description: Description of the JSON structure
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            
        Returns:
            Tuple of (parsed JSON dict or None, token usage dict)
        """
        temp = temperature if temperature is not None else self.default_temperature
        
        # Add instructions for JSON format
        json_instruction = f"""
Please provide the response strictly as a valid JSON object containing {description}.
Enclose the JSON object within triple backticks, like this:
```json
{{
  "key": "value",
  ...
}}
```
Respond ONLY with the JSON object within the backticks.
"""
        
        full_prompt = f"{prompt}\n{json_instruction}"
        
        self.logger.debug(
            f"Requesting completion (expecting JSON) from {self.model} "
            f"(max_tokens={max_tokens}, temp={temp})"
        )
        
        # Call standard completion method
        raw_text, usage_dict = await self.generate_completion(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temp
        )
        
        if not raw_text:
            self.logger.warning("LLM returned empty content when asked for JSON")
            return (None, usage_dict)
        
        # Parse JSON from response
        parsed_json = self._extract_and_parse_json(raw_text)
        
        if parsed_json is None:
            self.logger.warning(f"Failed to parse JSON from response: {raw_text[:200]}...")
        
        return (parsed_json, usage_dict)
    
    def generate_and_parse_json_sync(
        self,
        prompt: str,
        description: str = "the required data",
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], UsageDict]:
        """
        Generate text completion and parse JSON from response (sync).
        
        Args:
            prompt: User prompt text
            description: Description of the JSON structure
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            
        Returns:
            Tuple of (parsed JSON dict or None, token usage dict)
        """
        if not self._sync_client:
            self.logger.error("Synchronous client not initialized")
            return (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
        temp = temperature if temperature is not None else self.default_temperature
        
        # Add instructions for JSON format
        json_instruction = f"""
Please provide the response strictly as a valid JSON object containing {description}.
Enclose the JSON object within triple backticks, like this:
```json
{{
  "key": "value",
  ...
}}
```
Respond ONLY with the JSON object within the backticks.
"""
        
        full_prompt = f"{prompt}\n{json_instruction}"
        
        # Call standard completion method
        raw_text, usage_dict = self.generate_completion_sync(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temp
        )
        
        if not raw_text:
            self.logger.warning("LLM returned empty content when asked for JSON")
            return (None, usage_dict)
        
        # Parse JSON from response
        parsed_json = self._extract_and_parse_json(raw_text)
        
        if parsed_json is None:
            self.logger.warning(f"Failed to parse JSON from response")
        
        return (parsed_json, usage_dict)
    
    #
    # Enhanced Methods for Batch Processing
    #
    
    async def extract_evidence_batch(
        self,
        chunk_text: str,
        criteria_group: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], UsageDict]:
        """
        Extract evidence for a group of criteria from a text chunk.
        
        Args:
            chunk_text: Text chunk to analyze
            criteria_group: List of criteria to extract evidence for
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            Tuple of (evidence list, token usage dict)
        """
        # Create a prompt that efficiently asks for evidence for all criteria at once
        criteria_descriptions = []
        for i, criterion in enumerate(criteria_group):
            criterion_id = criterion.get("id", f"criterion_{i}")
            criterion_name = criterion.get("name", f"Criterion {i}")
            criterion_question = criterion.get("question", "")
            dimension_id = criterion.get("dimension_id", "unknown")
            dimension_name = criterion.get("dimension_name", "Unknown")
            
            criteria_descriptions.append(
                f"Criterion {i+1}: {criterion_name} (ID: {criterion_id})\n"
                f"Question: {criterion_question}\n"
                f"Dimension: {dimension_name} (ID: {dimension_id})"
            )
        
        criteria_text = "\n\n".join(criteria_descriptions)
        
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = """You are an expert evidence extractor. Your task is to analyze 
            document text and identify evidence relevant to multiple criteria. For each criterion, 
            extract text passages that provide direct evidence or insights. Only extract evidence 
            if it is truly relevant."""
        
        prompt = f"""Extract evidence from the following text chunk relevant to the criteria listed below.

TEXT CHUNK:
{chunk_text}

CRITERIA:
{criteria_text}

For each criterion where you find relevant evidence, include:
1. The criterion ID
2. The dimension ID
3. The extracted text (direct quote from the document)
4. A brief explanation of why this text is relevant
5. A confidence score (0.0-1.0) indicating how relevant this evidence is

Only include evidence for criteria where relevant content is found.
"""
        
        # Define schema for the evidence batch
        evidence_schema = {
            "type": "object",
            "properties": {
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion_id": {"type": "string"},
                            "dimension_id": {"type": "string"},
                            "text": {"type": "string"},
                            "relevance": {"type": "string"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["criterion_id", "dimension_id", "text"]
                    }
                }
            },
            "required": ["evidence"]
        }
        
        # Extract evidence using structured output
        result, usage = await self.generate_structured_output(
            prompt=prompt,
            output_schema=evidence_schema,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            description="evidence for multiple criteria"
        )
        
        # Process the result
        evidence_list = []
        if isinstance(result, dict) and "evidence" in result:
            evidence_list = result["evidence"]
        
        return (evidence_list, usage)
    
    async def process_parallel_queries(
        self,
        queries: List[Dict[str, Any]],
        process_fn: Callable,
        max_concurrent: int = 3
    ) -> List[Any]:
        """
        Process multiple queries in parallel with concurrency control.
        
        Args:
            queries: List of query parameters
            process_fn: Async function to process each query
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of results corresponding to queries
        """
        self.logger.info(f"Processing {len(queries)} queries with max_concurrent={max_concurrent}")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(query):
            async with semaphore:
                return await process_fn(query)
        
        # Create tasks for all queries
        tasks = [process_with_semaphore(query) for query in queries]
        
        # Execute tasks with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing query {i}: {str(result)}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results