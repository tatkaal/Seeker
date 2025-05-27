import openai
import json
import re

# --- 4. LLM Integrations ---
class LLMIntegrator:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        # Cache for LLM responses to save costs/time during development
        self.response_cache = {}


    def query_openai(self, prompt: str, system_prompt: str = "You are a helpful AI assistant specializing in job market analysis.", model: str = "gpt-3.5-turbo", max_tokens=500, temperature=0.5):
        if not self.openai_api_key:
            # print("OpenAI API key not set. Skipping OpenAI query.")
            return "OpenAI API key not set."
        
        cache_key = (prompt, model, max_tokens, temperature)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = response.choices[0].message.content.strip()
            self.response_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return f"OpenAI API error: {e}"

    def extract_structured_info_llm(self, text: str, info_schema: dict, model_openai="gpt-3.5-turbo"):
        """
        Extracts structured info from text based on a schema using an LLM.
        Schema example: {"skills": "list of skills", "experience_years": "number or range"}
        """
        prompt = f"""
        From the following job ad text, extract the information requested in the JSON schema provided below.
        If a piece of information is not found, use null or an empty list/string as appropriate for the type.
        Format the output as a single JSON object.

        JSON Schema:
        {json.dumps(info_schema, indent=2)}

        Job Ad Text:
        ---
        {text[:3000]} 
        ---
        Extracted JSON Output:
        """
        raw_response = self.query_openai(prompt, model=model_openai, max_tokens=1000, temperature=0.2)
        
        try:
            # LLMs sometimes add markdown ```json ... ``` or explanations. Try to clean it.
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
            if match:
                json_str = match.group(1)
            else:
                # Find the first '{' and last '}'
                first_brace = raw_response.find('{')
                last_brace = raw_response.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = raw_response[first_brace : last_brace+1]
                else:
                    json_str = raw_response # Fallback

            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"LLM JSON Decode Error. Raw response:\n{raw_response}")
            return None # Or return a dict with an error field

