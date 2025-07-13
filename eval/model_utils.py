import os
import time

import torch
from transformers import PreTrainedTokenizer
import functools
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import json

from typing import List, Tuple

# Import the new LocretCache implementation
from locret_cache import LocretCache

def load_attn_pattern_new(attn_load_dir, sink_size=None, recent_size=None):
    if attn_load_dir.endswith(".tsv"):
        path = attn_load_dir
    else:
        path = os.path.join(attn_load_dir, "full_attention_heads.tsv")
    full_attention_heads = np.loadtxt(
        path,
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    if sink_size is None:
        config = json.load(open(os.path.join(attn_load_dir, "config.json")))
        sink_size = config["sink_size"]
        recent_size = config["recent_size"]
    return full_attention_heads, sink_size, recent_size



## locret KV eviction

def locret_kv_eviction(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    scores: List[torch.Tensor],
    attentions: List[torch.Tensor],
    input_len: int,
    local_len: int,
    budget_size: int,
    stabilizers: int,
    start_index: int,
    end_index: int,
    n_layers: int,
):
    # Mostly taken verbatim from locret
    pruned_kv_cache = []
    kv_shape = past_key_values[0][0].shape
    for layer in range(n_layers):
        if len(scores) > layer:
            scores[layer] = torch.cat(
                (scores[layer], attentions[layer][:end_index-start_index]), # I don't know why we need to index here
                dim=-2
            )
        else:
            scores.append(
                attentions[layer][:end_index-start_index]
            )
        
        n_selected = min(budget_size, scores[layer].shape[-2])
        
        sc = scores[layer].clone()
        if end_index < input_len - local_len:
            sc[:, -stabilizers:, :] = torch.finfo(sc.dtype).max # always keep `stabilizers` last kvs if this is not the last chunk
        indices = torch.topk(sc[0, :, :], k=n_selected, dim=-2).indices 

        indices = indices.transpose(0, 1).sort().values # Sort the indices in ascending order
        scores[layer] = torch.gather(
            scores[layer],
            1,
            indices.transpose(0, 1).unsqueeze(0) # back to the original shape after sorting
        )            
        indices = indices.unsqueeze(0).unsqueeze(-1).repeat(kv_shape[0], 1, 1, kv_shape[3]) 
        k = torch.gather(past_key_values[layer][0], 2, indices)
        v = torch.gather(past_key_values[layer][1], 2, indices)
        pruned_kv_cache.append((k, v))
    
    return pruned_kv_cache, scores

## End locret

def format_chat(message, include_system=False, system_message="You are a helpful assistant."):
    if include_system:
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


def call_api(func, limit=5, pause=10):
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                output = None
                break
    return output


class LLM:
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    def prepare_inputs(self, test_item, data):
        raise NotImplementedError("prepare_inputs not implemented for LLM")
    
    def generate(self, inputs=None, prompt=None, **kwargs):
        raise NotImplementedError("generate not implemented for LLM")


class OpenAIModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ): 
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        import openai
        import tiktoken
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION 
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    
    def prepare_inputs(self, test_item, data):
        buffer = 100
        # we don't include system message to stay consistent with other models
        prompt = format_chat(data["user_template"].format(**test_item), include_system=False,)
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        max_length = self.max_length
        if max_length > 128000:
            logger.warning(f"max_length {max_length} is greater than 128000, setting to 128000")
            max_length = 128000

        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (max_length - self.generation_max_length - buffer)
            new_context = self.tokenizer.decode(self.tokenizer.encode(test_item["context"])[:-truncate_length])
            test_item["context"] = new_context
            prompt = format_chat(data["user_template"].format(**test_item), include_system=False)
        return prompt 

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """
    def generate(self, inputs=None, prompt=None, system_message="You are a helpful assistant", **kwargs):
        if inputs is None:
            inputs = format_chat(prompt, include_system=True, system_message=system_message)
        
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create, 
            model=self.model_name, 
            messages=inputs, 
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but sitll return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None

class AnthropicModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ): 
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        from anthropic import Anthropic, AnthropicVertex
        if "vertex" in model_name:
            # region defaults to env var CLOUD_ML_REGION and project_id defaults to ANTHROPIC_VERTEX_PROJECT_ID
            self.model = AnthropicVertex()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # remember to set ANTHROPIC_API_KEY environment variable (the default)
            self.model = Anthropic()

        self.tokenizer = self.model.get_tokenizer()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.do_sample = do_sample
        self.stops = None
        if stop_newline: # claude does not support newline
            pass


    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item), 
            include_system=False,
        )
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            tokens = self.tokenizer.encode(test_item["context"])
            new_context = test_item["context"][:tokens.offsets[-truncate_length-1][1]]
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item), 
                include_system=False,
            )
        return prompt
       

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """
    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = format_chat(prompt, include_system=False)
        
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        # Note: in the original paper, we used this system message:
        # system="You are a helpful assistant. Make sure your output does not contain new lines."
        # To be consistent with the other models, and for future compability, we remove the system message
        # We don't expect this to make a significant difference in the results
        func = functools.partial(
            self.model.messages.create,
            model=self.model_name, 
            messages=inputs, 
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop_sequences=self.stops,
            **kwargs,
        )
        output = call_api(func, pause=20)

        if output is not None:
            return {
                "output": output.content[0].text,
                "input_len": output.usage.input_tokens,
                "output_len": output.usage.output_tokens,
                "input_text": inputs,
            }
        return None


class GeminiModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ): 
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        import google.generativeai as genai
        # default env var GOOGLE_API_KEY
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        import vertexai
        vertexai.init() # make sure to set the env var appropriately
        from vertexai.preview.tokenization import get_tokenizer_for_model
        self.model = genai.GenerativeModel(model_name)
        self.tokenizer = get_tokenizer_for_model(model_name)
        self.model_name = model_name

    def prepare_inputs(self, test_item, data):
        prompt = data["prompt_template"].format(**test_item)
        buffer = 100
        inputs = self.tokenizer.compute_tokens(prompt).token_info_list[0].tokens
        input_len = len(inputs)

        max_length = self.max_length
        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (max_length - self.generation_max_length - buffer)
            # not the most pretty way of doing this but it works...
            # the documentation doesn't provide an official way to truncate
            new_context = self.tokenizer._sentencepiece_adapter._tokenizer.decode(self.tokenizer.compute_tokens(test_item["context"]).token_info_list[0].token_ids[:-truncate_length])
            test_item['context'] = new_context
            prompt = data["prompt_template"].format(**test_item)
        
        return prompt

    def generate(self, inputs=None, prompt=None, **kwargs):
        import google.generativeai as genai
        if inputs is None:
            inputs = prompt
        
        generation_config = genai.GenerationConfig(temperature=self.temperature, top_p=self.top_p, max_output_tokens=self.generation_max_length)
        func = functools.partial(
            self.model.generate_content, 
            contents=inputs,
            generation_config=generation_config
        )
        output = call_api(func, pause=15)
        if output is not None:
            try:
                # can probably check the output for errors but it's not well documented
                output.text
            except Exception as e:
                logger.error(f"Error in output: {output}; {e}")  
                return None

            return {
                "output": output.text,
                "input_len": output.usage_metadata.prompt_token_count,
                "output_len": output.usage_metadata.candidates_token_count,
                "input_text": inputs,
            }
        return None


class TogetherModel(LLM):
    def __init__(
        self,
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ):
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        from transformers import AutoTokenizer
        from together import Together
        # default env var TOGETHER_API_KEY
        self.model = Together()
        # should change this to be more flexible in the future lol
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
        self.model_name = model_name.replace("togetherapi/", "")
 
    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item), 
            system_message=data.get("system_message", "You are a helpful assistant.")
        )
        tokens = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        input_len = len(tokens)

        max_length = self.max_length
        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (max_length - self.generation_max_length - buffer)
            context_tokens = self.tokenizer(test_item["context"], return_offsets_mapping=True)
            new_context = test_item["context"][:context_tokens["offset_mapping"][-truncate_length][0]]
            
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item), 
                system_message=data.get("system_message", "You are a helpful assistant.")
            )
        return prompt 

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """
    def generate(self, inputs=None, prompt=None, system_message="You are a helpful assistant", **kwargs):
        if inputs is None:
            inputs = format_chat(prompt, include_system=True, system_message=system_message)
        
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create, 
            model=self.model_name, 
            messages=inputs, 
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but sitll return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None


def tokenize(sample, data, tokenizer, max_length, generation_max_length, use_chat_template=False):
    def format_input(sample):
        if use_chat_template:
            chat = format_chat(
                data["user_template"].format(**sample), 
                include_system=False,
                system_message=data.get("system_message", "You are a helpful assistant.")
            )
            try:
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                chat = format_chat(
                    data["user_template"].format(**sample), 
                    include_system=False,
                )
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            tokenized_input = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = data["prompt_template"].format(**sample)
            tokenized_input = tokenizer([prompt], return_tensors="pt")
        return tokenized_input
    
    if "Phi3SmallTokenizer" in str(type(tokenizer)):
        buffer = 64 if max_length == 131072 else 0 # there is some problem with their rotary emb implementation
    else:
        buffer = 0
    
    tokenized_input = format_input(sample)
    if tokenized_input.input_ids.size(1) > max_length - generation_max_length - buffer:
        truncate_length = tokenized_input.input_ids.size(1) - (max_length - generation_max_length - buffer)

        # handle non-fast hf tokenizers (e.g., phi-3-small)
        if isinstance(tokenizer, PreTrainedTokenizer) and not tokenizer.is_fast:
            context_tokens = tokenizer(sample["context"])
            new_context = tokenizer.decode(context_tokens["input_ids"][:-truncate_length])
        else:
            context_tokens = tokenizer([sample["context"]], return_offsets_mapping=True)
            new_context = sample["context"][:context_tokens["offset_mapping"][0][-truncate_length][0]]

        sample["context"] = new_context
        tokenized_input = format_input(sample)
    return tokenized_input


class HFModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
        **kwargs,
    ):
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
        
        model_kwargs = {}
        from pkg_resources import parse_version
        if parse_version(transformers.__version__) <= parse_version("4.34.1"):
            model_kwargs["use_flash_attention_2"] = True
        else:
            model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        if "recurrentgemma" in model_name or "yarn" in model_name.lower():
            model_kwargs = {}

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "rope_theta" in kwargs and kwargs["rope_theta"] is not None:
            logger.info(f"Override rope theta to {kwargs['rope_theta']}")
            config.rope_theta = kwargs["rope_theta"]
        
        if "locret_bin_file" in kwargs and kwargs["locret_bin_file"] is not None:
            from locret.models.llama.modeling_llama import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                device_map="auto",
                trust_remote_code=True,
            )
            ckpt = torch.load(kwargs["locret_bin_file"])
            pruned_ckpt = {}
            for k, v in ckpt.items():
                if 'fc' in k:
                    pruned_ckpt[k] = v
            self.model.load_state_dict(pruned_ckpt, strict=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                config=config,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                device_map="auto",
                trust_remote_code=True,
                **model_kwargs
            )

        self.special_settings = {}
        if "duoattn" in kwargs and kwargs["duoattn"] is not None:
            logger.warning("Using DuoAttention patch for evaluation!")
            logger.warning("Note that when using DuoAttention, we use eager attention implementation for compatibility")
            from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
            from duo_attn.patch import enable_duo_attention_eval
            duoattn_path = kwargs["duoattn"]
            duoattn_sparsity = kwargs["duoattn_sparsity"]
            attn_heads, sink_size, recent_size = load_attn_pattern_new(
                duoattn_path,
                sink_size=kwargs["duoattn_sink"],
                recent_size=kwargs["duoattn_sliding"],
            )
            if kwargs["duoattn_flipping"]:
                logger.warning("| Flipping the duoattn pattern (for debugging purposes)")
                attn_heads = 1 - attn_heads

            attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=duoattn_sparsity)
            enable_duo_attention_eval(
                self.model,
                attn_heads,
                sink_size=kwargs["duoattn_sink"],
                recent_size=kwargs["duoattn_sliding"],
            )
            self.chunk_prefilling = kwargs["duoattn_chunk_prefilling"]
            if self.chunk_prefilling is not None:
                logger.warning(f"Using chunk prefilling (size={self.chunk_prefilling}) for DuoAttention!")

            self.special_settings["method"] = "duo"
            self.special_settings["method_params"] = {
                "sparsity": duoattn_sparsity,
                "sink_size": kwargs["duoattn_sink"],
                "recent_size": kwargs["duoattn_sliding"],
            }
        elif "minference" in kwargs and kwargs["minference"] is not None:
            from minference import MInference
            from minference.modules.kvcompression import method_to_cache_obj
            print(f"**** USING {kwargs['minference']} for evaluation! ****")
            if kwargs["minference"] == "minference":
                logger.warning("Using MInference for evaluation!")
                minference = MInference(attn_type="minference", model_name=kwargs["minference_model_name"])
                self.model, _ = minference(self.model)
                raise NotImplementedError("MInference is not supported!")
            elif kwargs["minference"] in ["pyramidkv", "snapkv"]:
                logger.warning("Using PyramidKV for evaluation!")
                minference = MInference(
                    attn_type="dense",
                    kv_type=kwargs["minference"],
                    model_name=kwargs["minference_model_name"],
                    attn_kwargs={
                        "window_size": kwargs["minference_window_size"],
                        "max_capacity_prompt": kwargs["minference_max_capacity_prompt"],
                        "compress_group_kvs": kwargs["minference_compress_group_kvs"],
                    },
                )
                self.model, config = minference(self.model)

                # Ready a cache
                config.num_layers = self.model.config.num_hidden_layers
                self.special_settings["past_key_values"] = (method_to_cache_obj[kwargs["minference"]], config)
                self.special_settings["is_pyramid_snapkv"] = True

                # If we have a sparsity value, use that
                if kwargs["minference_sparsity"] is not None:
                    self.special_settings["sparsity"] = kwargs["minference_sparsity"]
                else:
                    self.special_settings["total_prefill_budget"] = kwargs["minference_max_capacity_prompt"] # - kwargs["minference_window_size"]? Maybe not
                self.special_settings["local_window_size"] = kwargs["minference_window_size"]
                self.special_settings["method"] = "pyramid_snap_kv" 
                self.special_settings["method_params"] = {
                    "window_size": kwargs["minference_window_size"],
                    "max_capacity_prompt": kwargs["minference_max_capacity_prompt"],
                    "do_patch": kwargs.get("minference_chunking_patch", False),
                    "compress_group_kvs": kwargs["minference_compress_group_kvs"],
                }
            elif kwargs["minference"] == "l2":
                logger.warning("Using L2 for evaluation!")
                minference = MInference(
                    attn_type="dense",
                    kv_type=kwargs["minference"],
                    model_name=kwargs["minference_model_name"],
                    attn_kwargs={
                        "num_skip_layers": 2,
                        "max_capacity_total": kwargs["minference_max_capacity_prompt"],
                        "num_local_tokens": kwargs["minference_window_size"],
                    }
                )
                self.model, config = minference(self.model)

                # Ready a cache
                config.num_layers = self.model.config.num_hidden_layers
                self.special_settings["past_key_values"] = (method_to_cache_obj[kwargs["minference"]], config)
                self.special_settings["method"] = "l2"
                self.special_settings["method_params"] = {
                    "num_skip_layers": 2,
                    "num_total_layers": self.model.config.num_hidden_layers,
                    "max_capacity_total": kwargs["minference_max_capacity_prompt"],
                }
            else:
                raise ValueError(f"Invalid minference type: {kwargs['minference']}")
            
            self.chunk_prefilling = kwargs.get("minference_chunk_prefilling", None)            
        elif "locret_bin_file" in kwargs and kwargs["locret_bin_file"] is not None:
            logger.warning("Using Locret for evaluation!")
            self.special_settings["method"] = "locret"
            self.special_settings["method_params"] = {
                "sparsity": kwargs["locret_sparsity"],
                "budget_size": kwargs["locret_budget_size"],
                "local_len": kwargs["locret_local_len"],
                "stabilizers": kwargs["locret_stabilizers"],
            }
            
            # Set up LocretCache configuration - this will be used to create cache instances
            self.special_settings["locret_cache_config"] = {
                "budget_size": kwargs["locret_budget_size"],
                "local_len": kwargs["locret_local_len"], 
                "stabilizers": kwargs["locret_stabilizers"],
                "num_layers": None,  # Will be set during generation
            }

            # Locret never has a `None` chunk_prefilling
            # On the upper end, use 128k
            self.chunk_prefilling = kwargs.get("locret_chunk_prefilling", 131072)
        else:
            self.chunk_prefilling = None

        if kwargs.get("torch_compile", True):
            logger.warning("Using torch compile for evaluation!")
            self.model = torch.compile(self.model)

        # use the default if possible, append if necessary
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
            if "llama" in model_name.lower():
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device
        self.disable_prefill = False

        if "gemma" in model_name.lower():
            self.disable_prefill = True
            logger.warning("gemma models cannot prefill with past kvs due to cache implementation, need to change the code manually if you need to prefill")
    
    
    def prepare_inputs(self, test_item, data):
        return tokenize(
            test_item, 
            data, 
            tokenizer=self.tokenizer, 
            max_length=self.max_length,
            generation_max_length=self.generation_max_length,
            use_chat_template=self.use_chat_template,
        )
    
    
    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = self.tokenizer([prompt], return_tensors="pt", max_length=self.max_length-self.generation_max_length, truncation=True, padding=True)
        
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        if hasattr(self.model, "model") and not self.disable_prefill:
            # prefill without calculating the logits (save memory for large vocab models)
            extra = {}
            if "jamba" in str(type(self.model)).lower():
                from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache
                cache = HybridMambaAttentionDynamicCache(self.model.config, inputs.input_ids.shape[0], self.model.dtype, device=self.model.device)
                extra = {"past_key_values": cache}

            # We should do chunked prefilling if (1) there is a valid chunk prefilling size, or (2) if we have a special setting that tells us that the last `k` tokens should be prefilled separately
            method = self.special_settings.get("method", "vanilla")
            if method == "locret":
                extra["output_attentions"] = True

            if self.chunk_prefilling is not None:
                past_key_values = None
                prefilling_input_ids = inputs.input_ids[..., :-1]

                if method == "locret":
                    # Initialize LocretCache for locret
                    cache_config = self.special_settings["locret_cache_config"].copy()
                    cache_config["num_layers"] = self.model.config.num_hidden_layers
                    past_key_values = LocretCache(**cache_config)
                    
                    # For locret, the last `local_len` tokens are considered their own chunk
                    if input_len-1 <= self.special_settings["method_params"]["local_len"]:
                        # A single chunk
                        prefill_indices = [0]
                        prefill_sizes = [input_len-1]
                    else:
                        # Multiple chunks
                        prefill_to = (input_len-1) - self.special_settings["method_params"]["local_len"]
                        prefill_indices = list(
                            range(
                                0, 
                                prefill_to, 
                                self.chunk_prefilling
                            )
                        )
                        
                        prefill_sizes = [self.chunk_prefilling] * (len(prefill_indices) - 1)
                        prefill_sizes.append(prefill_to - self.chunk_prefilling * (len(prefill_indices) - 1))

                        prefill_indices.append(prefill_to)
                        prefill_sizes.append(self.special_settings["method_params"]["local_len"])
                else:
                    prefill_indices = list(range(0, prefilling_input_ids.size(1), self.chunk_prefilling))
                    prefill_sizes = [self.chunk_prefilling] * (len(prefill_indices) - 1)
                    prefill_sizes.append(prefilling_input_ids.size(1) - self.chunk_prefilling * (len(prefill_indices) - 1))

                # Prefill
                for i, (index, size) in enumerate(zip(prefill_indices, prefill_sizes)):
                    # For pyramid/snapkv: (1) use the readied cache at the beginning, and (2) apply the special settings
                    if past_key_values is None and "past_key_values" in self.special_settings:
                        past_key_values_class, past_key_values_config = self.special_settings["past_key_values"]
                        past_key_values = past_key_values_class(past_key_values_config)
                    if self.special_settings.get("is_pyramid_snapkv", False):                        
                        # Now, apply the special settings
                        # The chunks are not equally sized, so recalculate budget for this chunk
                        prefill_budget = int((1 - self.special_settings["sparsity"]) * size)
                        if (
                            self.special_settings["method_params"]["do_patch"] and
                            index + size < prefilling_input_ids.size(1)
                        ):
                            # We will patch an extra window_size tokens, so we need to add that to the prefill budget
                            prefill_budget += self.special_settings["method_params"]["window_size"]

                        past_key_values.apply_special(
                            is_prefill=True, 
                            capacity_override=prefill_budget
                        )

                    chunk = prefilling_input_ids[:, index : index + size]
                    # If (1) we do patching and (2) this is not the last chunk, then we need to:
                    # (a) append the last window_size tokens from prefilling_input_ids to the chunk
                    # (b) run the forward pass and let everything be cached
                    # (c) remove the last window_size tokens from the chunk and the past_key_values
                    # This makes sure that caching decisions use the actual last window_size tokens
                    if (
                        "method_params" in self.special_settings and
                        "do_patch" in self.special_settings["method_params"] and
                        self.special_settings["method_params"]["do_patch"] and
                        index + size < prefilling_input_ids.size(1)
                    ):
                        # (a)
                        chunk = torch.cat(
                            [
                                chunk,
                                prefilling_input_ids[:, -self.special_settings["method_params"]["window_size"] :],
                            ],
                            dim=1
                        )              
                    
                    # print(f"Prefilling {index} -> {index + size} of {prefilling_input_ids.size(1)}")
                    output = self.model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, **extra)
                    past_key_values = output.past_key_values

                    # (c)
                    if (
                        "method_params" in self.special_settings and
                        "do_patch" in self.special_settings["method_params"] and
                        self.special_settings["method_params"]["do_patch"] and
                        index + size < prefilling_input_ids.size(1)
                    ):
                        past_key_values.drop_last_k_tokens(self.special_settings["method_params"]["window_size"])
                    elif method == "locret":
                        # Perform kv eviction using the new LocretCache interface
                        # But first get the budget
                        if "sparsity" in self.special_settings["method_params"]:
                            # The budget dynamically increases, and is given by (1 - sparsity) * current_total_len
                            budget = int(
                                (1 - self.special_settings["method_params"]["sparsity"]) * (index + size)
                            )
                        else:
                            budget = self.special_settings["method_params"]["budget_size"]
                        # Round up to nearest multiple of 64
                        budget = ((budget + 63) // 64) * 64
                        # Budget should at least be the number of stabilizers
                        budget = max(budget, self.special_settings["method_params"]["stabilizers"])

                        # Apply eviction using the new cache interface
                        past_key_values.apply_locret_eviction(
                            attentions=output.attentions,
                            input_len=input_len-1, # the total number of tokens we will prefill
                            start_index=index,
                            end_index=index + size,
                            budget_size=budget, # increases dynamically
                        )

                if self.special_settings.get("is_pyramid_snapkv", False) and past_key_values is not None:
                    # We are no longer prefilling
                    past_key_values.reset_special()

                inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "past_key_values": past_key_values}
            else: 
                if "past_key_values" in self.special_settings:
                    past_key_values_class, past_key_values_config = self.special_settings["past_key_values"]
                    past_key_values = past_key_values_class(past_key_values_config)

                if self.special_settings.get("is_pyramid_snapkv", False):                        
                    # The entire KV budget goes to this chunk
                    if "sparsity" in self.special_settings:
                        capacity_override = int((1 - self.special_settings["sparsity"]) * inputs.input_ids.size(1))
                    else:   
                        capacity_override = self.special_settings["total_prefill_budget"]

                    # Round up to nearest multiple of 64 and take max with the local_window_size
                    capacity_override = max(capacity_override, self.special_settings["local_window_size"])
                    capacity_override = ((capacity_override + 63) // 64) * 64
                    past_key_values.apply_special(
                        is_prefill=True, 
                        capacity_override=capacity_override
                    )
                    extra["past_key_values"] = past_key_values

                # We don't need to worry about any patching here - the last few tokens are included in the prefill
                prefill = self.model.model(
                    input_ids=inputs.input_ids[..., :-1], 
                    attention_mask=inputs.attention_mask[..., :-1], 
                    **extra
                )
                past_key_values = prefill.past_key_values 

                if self.special_settings.get("is_pyramid_snapkv", False) and past_key_values is not None:
                    # We are no longer prefilling
                    past_key_values.reset_special()

                inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "past_key_values": past_key_values}
                if past_key_values is None:
                    self.disable_prefill = True
                    logger.warning("past key values is None, not able to prefill with KVs, disabling...")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        
        text = self.tokenizer.decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)
        save_prompt = self.tokenizer.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.tokenizer.decode(inputs["input_ids"][0][-500:])

        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1) - input_len,
            "input_text": save_prompt,
        }



class VLLMModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
    ):
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        
        from vllm import LLM
        # at the time of testing: note that the max model length is derived from the config file, and if max_length is larger than that length, there will be an error. it appears that vllm does not support positional extrapolation
        # there are some work arounds to this, but it may give unexpected results. 
        self.model = LLM(
            model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            trust_remote_code=True,
            # enforce_eager=True,
        )
        self.tokenizer = self.model.get_tokenizer()


    def prepare_inputs(self, test_item, data):
        return tokenize(
            test_item, 
            data, 
            tokenizer=self.tokenizer, 
            max_length=self.max_length,
            generation_max_length=self.generation_max_length,
            use_chat_template=self.use_chat_template,
        )
    

    def generate(self, inputs=None, prompt=None, **kwargs):
        from vllm import SamplingParams, TokensPrompt
        if inputs is None:
            inputs = self.tokenizer([prompt], return_tensors="pt", max_length=self.max_length-self.generation_max_length, truncation=True, padding=True)
        
        self.sampling_params = SamplingParams(
            temperature = self.temperature if self.do_sample else 0.0,
            top_p = self.top_p,
            max_tokens = self.generation_max_length,
        )

        outputs = self.model.generate(
            prompts=TokensPrompt(prompt_token_ids=inputs["input_ids"][0].tolist()),
            sampling_params=self.sampling_params,
            **kwargs
        )[0]
        save_prompt = self.tokenizer.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.tokenizer.decode(inputs["input_ids"][0][-500:])
        return {
            "output": outputs.outputs[0].text,
            "input_len": len(outputs.prompt_token_ids),
            "output_len": len(outputs.outputs[0].token_ids),
            "input_text": save_prompt,
        }


def load_LLM(args):
    kwargs = {}
    if "gpt" in args.model_name_or_path:
        model_cls = OpenAIModel
    elif "claude" in args.model_name_or_path:
        model_cls = AnthropicModel
    elif "gemini" in args.model_name_or_path:
        model_cls = GeminiModel
    elif "togetherapi" in args.model_name_or_path:
        model_cls = TogetherModel
    elif args.use_vllm:
        model_cls = VLLMModel
    else:
        model_cls = HFModel
        if args.no_torch_compile:
            kwargs["torch_compile"] = False
        if args.no_bf16:
            kwargs["torch_dtype"] = torch.float32
        if args.rope_theta is not None:
            kwargs["rope_theta"] = args.rope_theta

        if args.duoattn is not None:
            kwargs["duoattn"] = args.duoattn
            kwargs["duoattn_sparsity"] = args.duoattn_sparsity
            kwargs["duoattn_sink"] = args.duoattn_sink
            kwargs["duoattn_sliding"] = args.duoattn_sliding
            kwargs["duoattn_chunk_prefilling"] = args.duoattn_chunk_prefilling
            kwargs["duoattn_flipping"] = args.duoattn_flipping
            kwargs["attn_implementation"] = "eager"
        
        if args.minference is not None:
            kwargs["minference"] = args.minference  
            kwargs["minference_model_name"] = args.minference_model_name
            kwargs["minference_chunk_prefilling"] = args.minference_chunk_prefilling
            kwargs["minference_window_size"] = args.minference_window_size
            kwargs["minference_max_capacity_prompt"] = args.minference_max_capacity_prompt
            kwargs["minference_sparsity"] = args.minference_sparsity
            kwargs["minference_chunking_patch"] = args.minference_chunking_patch
            kwargs["minference_grouped_eviction"] = args.minference_grouped_eviction
            kwargs["minference_compress_group_kvs"] = args.minference_compress_group_kvs

        if args.locret_bin_file is not None:
            kwargs["locret_bin_file"] = args.locret_bin_file
            kwargs["locret_sparsity"] = args.locret_sparsity
            kwargs["locret_budget_size"] = args.locret_budget_size
            kwargs["locret_local_len"] = args.locret_local_len
            kwargs["locret_stabilizers"] = args.locret_stabilizers
            kwargs["locret_chunk_prefilling"] = args.locret_chunk_prefilling
            
    model = model_cls(
        args.model_name_or_path, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_length=args.input_max_length, 
        generation_max_length=args.generation_max_length, 
        generation_min_length=args.generation_min_length, 
        do_sample=args.do_sample, 
        stop_newline=args.stop_newline, 
        use_chat_template=args.use_chat_template,
        **kwargs,
    )

    return model