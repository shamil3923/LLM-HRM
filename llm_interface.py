"""
LLM interface module.
Provides a generic call_llm function for LLM interaction.
"""

import os
from openai import OpenAI

# Initialize OpenAI client
# Set your API key as an environment variable: export OPENAI_API_KEY="your-key-here"
# Or pass it directly: client = OpenAI(api_key="your-key-here")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"))


def call_llm(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 1000) -> str:
    """
    Generic function to call an LLM with a text prompt.
    
    Args:
        prompt: The text prompt to send to the LLM
        model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in response
        
    Returns:
        The LLM's text response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise mathematical reasoning assistant. Follow instructions exactly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
