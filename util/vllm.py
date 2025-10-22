from dataclasses import asdict
import json
from typing import List, Dict, Optional
import aiohttp
import asyncio
import vllm
from pandas import DataFrame
from vllm import EngineArgs, LLM as LLMEntrypoint, SamplingParams
from util.utils import post_http_request, is_server_running, get_tokenizer, async_post_http_request
from util.prompt import DEFAULT_SYSTEM_PROMPT
from util.utils import LLM
import re

class vLLM(LLM):
    def __init__(self, engine_args: EngineArgs, sampling_params: Optional[SamplingParams] = None, base_url: Optional[str] = None):
        self.base_url = None
        if base_url:
            model = is_server_running("http://localhost:8000/v1/models")
            if model:
                self.base_url = base_url
                self.model = model
                self.tokenizer = get_tokenizer()
                print(f"Connected to vLLM server running at {self.base_url}")
            else:
                print("vLLM server connection error")

    def _generate_prompt(self, user_prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": system_prompt}
        ]

        successful_prompt_generation = False
        while not successful_prompt_generation:
            try:
                # Construct a prompt for the chosen model given OpenAI style messages.
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                if messages[0]["role"] == "system":
                    # Try again without system prompt
                    messages = messages[1:]
                else:
                    raise e
            else:
                successful_prompt_generation = True
        
        return prompt
    
    # TODO: Few-shot examples
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        fields_json = json.dumps(fields)

        user_prompt = f"Answer the below query:\n\n{query}\n\ngiven the following data:\n\n{fields_json}"
        
        prompt = self._generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt)
        if self.base_url:
            output = json.loads(post_http_request(self.model, [prompt], temperature=0, api_url=(self.base_url + "/completions")).content)
        else:
            output = self.engine.generate(prompts=[prompt], sampling_params=self.sampling_params, use_tqdm=False)
        assert len(output) == 1
        return output[0].outputs[-1].text

    def split_into_chunks(self, lst: List, n: int) -> List[List]:
        """Split a list into roughly equal n chunks"""
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    async def execute_batch_async(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, guided_choice: List[str] = None) -> List[str]: 
        """Async batched execution with 10 parallel requests"""
        # Original prompt generation logic
        fields_json_list = [json.dumps(field) for field in fields]
        user_prompt_template = f"Given the following data:\n {{fields_json}} \n answer the below query:\n{query}"
        user_prompts = [user_prompt_template.replace("{{fields_json}}", fj) for fj in fields_json_list]
        prompts = [self._generate_prompt(up, system_prompt) for up in user_prompts]

        # Split prompts into 10 chunks
        chunks = self.split_into_chunks(prompts, 20)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for chunk in chunks:
                if chunk:  # Skip empty chunks
                    task = async_post_http_request(
                        session=session,
                        model=self.model,
                        prompts=chunk,
                        api_url=f"{self.base_url}/completions",
                        temperature=0,
                        guided_choice=guided_choice,
                    )
                    tasks.append(task)
            
            # Gather all responses
            responses = await asyncio.gather(*tasks)
            
            # Merge results while preserving order
            all_choices = []
            for response in responses:
                all_choices.extend(response.get('choices', []))
            
            return [choice['text'] for choice in all_choices]

    def execute_batch(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, guided_choice: List[str] = None) -> List[str]:
        return asyncio.run(self.execute_batch_async(fields, query, system_prompt, guided_choice))

    def execute_batch_sync(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, guided_choice: List[str] = None) -> List[str]:
        """Batched version of `execute`."""

        fields_json_list = [json.dumps(field) for field in fields]
        user_prompt_template = "Given the following data:\n {{fields_json}} \n answer the below query:\n"
        user_prompt_template += query

        user_prompts = [user_prompt_template.replace("{{fields_json}}", fields_json) for fields_json in fields_json_list]

        prompts = [self._generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) for user_prompt in user_prompts]
        if self.base_url:
            request_outputs = json.loads(post_http_request(self.model, prompts, temperature=0, api_url=(self.base_url + "/completions"), guided_choice=guided_choice).content)
            return [choice['message']['content'] for choice in request_outputs['choices']]
        else:
            request_outputs = self.engine.generate(prompts=prompts, sampling_params=self.sampling_params)
            return [output for output in request_outputs]

    def execute_batch_v2(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, guided_choice: List[str] = None) -> List[str]:        
        fields_json_list = [json.dumps(field) for field in fields]
        user_prompt_template = "Given the following data:\n {{fields_json}} \n answer the below query:\n"
        user_prompt_template += query

        user_prompts = [user_prompt_template.replace("{{fields_json}}", fields_json) for fields_json in fields_json_list]
        prompts = [self._generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) for user_prompt in user_prompts]
        print("len of prompts" + str(len(prompts)))
        outputs = []
        if self.base_url:
            # For each prompt, send a separate HTTP POST request and collect its answer.
            for prompt in prompts:
                response = post_http_request(self.model, [prompt], temperature=0, api_url=(self.base_url + "/completions"), guided_choice=guided_choice)
                request_output = json.loads(response.content)
                # Pick the corresponding answer out of the returned choices
                choices = request_output.get('choices', [])
                if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                    outputs.append(choices[0]['message']['content'])
                else:
                    outputs.append(None)  # or append "" or raise Exception, depending on error handling desired
            return outputs
        else:
            request_outputs = self.engine.generate(prompts=prompts, sampling_params=self.sampling_params)
            return [output for output in request_outputs]

        



