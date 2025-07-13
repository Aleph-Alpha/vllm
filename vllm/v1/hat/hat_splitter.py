# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import re
from abc import ABC, abstractmethod
from typing import Optional

from hat_splitter import HATSplitter as Splitter


class HATSplitter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> list[list[int]]:
        pass

    @abstractmethod
    def encode_to_flattened(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self,
               token_ids: list[int],
               errors: str = "replace",
               skip_special_tokens: bool = False) -> str:
        pass


class HATRuleSplitter(HATSplitter):

    def __init__(self,
                 special_token_dict: dict | None = None,
                 max_word_size: int = 100):
        super().__init__()
        self.hat_splitter = Splitter()
        self.max_word_size = max_word_size
        self.special_token_dict = special_token_dict
        self.special_token_replace: dict[int, list[int]] = {
            token: list(text.encode("utf-8"))
            for text, token in self.special_token_dict.items()
        }
        self.special_token_pattern = (re.compile(
            rf"({'|'.join(map(re.escape, special_token_dict.keys()))})")
                                      if special_token_dict else
                                      re.compile(r"(?!)"))
        self.eot_id: Optional[int] = self.special_token_dict.get('<|eot_id|>')
        assert self.eot_id, "eot_id not found in special_token_dict"

    def encode(self, text: str) -> list[list[int]]:
        chunks = []
        for str_chunk in self.special_token_pattern.split(text):
            if str_chunk:
                if str_chunk in self.special_token_dict:
                    chunks.append([self.special_token_dict[str_chunk]])
                else:
                    chunks.extend(
                        list(chunk)
                        for chunk in self.hat_splitter.split_with_limit(
                            str_chunk, self.max_word_size))
        return chunks

    def encode_to_flattened(self, text: str) -> list[int]:
        nested_chunks = self.encode(text)
        return list(itertools.chain.from_iterable(nested_chunks))

    def decode(self,
               token_ids: list[int],
               errors: str = "replace",
               skip_special_tokens: bool = False) -> str:
        new_token_ids: list[int]
        if skip_special_tokens:
            new_token_ids = [
                token_id for token_id in token_ids
                if token_id not in self.special_token_replace
            ]
        else:
            new_token_ids = []
            for token in token_ids:
                if token in self.special_token_replace:
                    new_token_ids.extend(self.special_token_replace[token])
                else:
                    new_token_ids.append(token)
        return bytes(new_token_ids).decode("utf-8", errors=errors)
