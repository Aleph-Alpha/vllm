# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from jinja2.sandbox import ImmutableSandboxedEnvironment

from vllm.transformers_utils.tokenizer_base import TokenizerBase
from vllm.utils import is_list_of
from vllm.v1.hat.hat_splitter import HATRuleSplitter


@dataclass
class Encoding:
    input_ids: Union[List[int], List[List[int]]]


@dataclass(unsafe_hash=True)
class HATTokenizer(TokenizerBase):

    def __init__(self, special_token_dict: Dict[str, int]):
        self.hat_splitter = HATRuleSplitter(special_token_dict)
        self.name_or_path = "HAT"
        self.jinja2_env = ImmutableSandboxedEnvironment()
        self.special_tokens_map = None

    @property
    def all_special_tokens_extended(self) -> List[str]:
        raise NotImplementedError()

    @property
    def all_special_tokens(self) -> List[str]:
        raise NotImplementedError()

    @property
    def all_special_ids(self) -> List[int]:
        raise NotImplementedError()

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 192

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return 256

    @property
    def max_token_id(self) -> int:
        return 255

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        raise NotImplementedError()

    def __call__(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        input_ids: Union[List[int], List[List[int]]]
        # For List[str], original prompt text
        if is_list_of(text, str):
            input_ids_: List[List[int]] = []
            for p in text:
                each_input_ids = self.encode_one(p, truncation, max_length)
                input_ids_.append(each_input_ids)
            input_ids = input_ids_
        # For List[int], apply chat template output, already tokens.
        elif is_list_of(text, int):
            input_ids = text
        # For str, single prompt text
        else:
            input_ids = self.encode_one(text, truncation, max_length)
        return Encoding(input_ids=input_ids)

    def get_vocab(self) -> Dict[str, int]:
        return {'a': i for i in range(256)}

    def get_added_vocab(self) -> Dict[str, int]:
        return {'a': i for i in range(256)}

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        input_ids = self.encode(text, truncation, max_length)

        if truncation:
            input_ids = input_ids[:max_length]
        return input_ids

    def encode(self,
               text: str,
               truncation: Optional[bool] = False,
               max_length: Optional[int] = None,
               add_special_tokens: Optional[bool] = None) -> List[int]:
        input_ids = self.hat_splitter.encode_to_flattened(text)

        if truncation:
            input_ids = input_ids[:max_length]
        return input_ids

    def encode_to_words(self, text: str) -> List[List[int]]:
        return self.hat_splitter.encode(text)

    def apply_chat_template(self,
                            conversation,
                            chat_template: str,
                            tokenize: bool,
                            tools: Optional[List[Dict[str, Any]]] = None,
                            **kwargs) -> str:
        compiled_template = self.jinja2_env.from_string(chat_template)
        rendered = compiled_template.render(messages=conversation,
                                            add_generation_prompt=True)
        return rendered

    def get_chat_template(self,
                          chat_template: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None) -> str:
        # Only needed for some tests where the system prompt is not included.
        #return "{%- set loop_messages = messages -%}{%- for message in loop_messages -%}{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] | trim + '<|eot_id|>' -%}{%- if loop.index0 == 0 -%}{%- set content = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|>' + content -%}{%- endif -%}{{- content -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|start_header_id|>assistant<|end_header_id|>' -}}{%- endif -%}"
        # We might need \n at the end of each message.
        return "{%- set loop_messages = messages -%}{%- for message in loop_messages -%}{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] | trim + '<|eot_id|>' -%}{%- if loop.index0 == 0 -%}{%- set content = '<|begin_of_text|>' + content -%}{%- endif -%}{{- content -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|start_header_id|>assistant<|end_header_id|>' -}}{%- endif -%}"

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.decode(int(token) for token in tokens)

    def decode(self,
               ids: Union[List[int], int],
               skip_special_tokens: bool = True) -> str:
        if isinstance(ids, int):
            ids = [ids]

        return self.hat_splitter.decode(ids)

    def convert_ids_to_tokens(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        # For byte models, tokens cannot be well defined.
        return [str(id_) for id_ in ids]
