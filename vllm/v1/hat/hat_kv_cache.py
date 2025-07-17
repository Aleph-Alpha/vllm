# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Callable, Dict, List, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import sha256, sha256_cbor_64bit
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager, get_manager_for_kv_cache_spec)
from vllm.v1.hat.hat_splitter import HATRuleSplitter
from vllm.v1.hat.hat_utils import HATKVCacheState, split_text
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        SlidingWindowSpec)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


class HATKVCacheManager(KVCacheManager):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = (
            sha256_cbor_64bit if caching_hash_algo == "sha256_cbor_64bit" else
            sha256 if caching_hash_algo == "sha256" else hash)
        init_none_hash(self.caching_hash_fn)
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = HATKVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=enable_caching,
            caching_hash_fn=self.caching_hash_fn,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHash]] = defaultdict(list)

        self.req_id_to_hat_info: Dict[str, HATKVCacheState] = {}
        self.hat_splitter = HATRuleSplitter(
            special_token_dict=vllm_config.model_config.hf_config.
            special_token_dict,
            max_word_size=vllm_config.model_config.hf_config.max_word_size)

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        return self.create_empty_block_list(), 0

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_draft_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = [
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups))
            ]

        req_id = request.request_id
        new_slots_needed_backbone = 0
        copy_state = None
        if req_id not in self.req_id_to_hat_info:
            self.req_id_to_hat_info[req_id] = HATKVCacheState(
                num_curr_word_bytes=0,
                num_computed_tokens_backbone=0,
                num_computed_tokens_byte=0)
            words = split_text(self.hat_splitter,
                               request._all_token_ids[:num_new_tokens])
            # Allocate a slot for the incomplete word here, so we are always one step ahead
            new_slots_needed_backbone = len(words)

            self.req_id_to_hat_info[
                request.request_id].num_curr_word_bytes = len(words[-1])
            self.req_id_to_hat_info[
                request.
                request_id].num_computed_tokens_backbone = new_slots_needed_backbone
            self.req_id_to_hat_info[
                request.request_id].num_computed_tokens_byte = num_new_tokens

        else:
            copy_state = HATKVCacheState(
                num_curr_word_bytes=self.req_id_to_hat_info[request.request_id].num_curr_word_bytes,
                num_computed_tokens_backbone=self.req_id_to_hat_info[request.request_id].num_computed_tokens_backbone,
                num_computed_tokens_byte=self.req_id_to_hat_info[request.request_id].num_computed_tokens_byte)
            if len(request._all_token_ids) - request.num_computed_tokens == 1:
                num_new_tokens = len(request._all_token_ids) - self.req_id_to_hat_info[request.request_id].num_computed_tokens_byte
            start_idx = self.req_id_to_hat_info[request.request_id].num_computed_tokens_byte - self.req_id_to_hat_info[request.request_id].num_curr_word_bytes
            offset = num_new_tokens + self.req_id_to_hat_info[request.request_id].num_curr_word_bytes
            words = split_text(
                self.hat_splitter,
                request._all_token_ids[start_idx:start_idx + offset])

            if len(words) > 1:
                self.req_id_to_hat_info[
                    request.request_id].num_curr_word_bytes = len(words[-1])
                new_slots_needed_backbone = len(words) - 1
                self.req_id_to_hat_info[
                    request.
                    request_id].num_computed_tokens_backbone += new_slots_needed_backbone
            else:
                self.req_id_to_hat_info[
                    request.request_id].num_curr_word_bytes = len(words[0])
            self.req_id_to_hat_info[
                request.request_id].num_computed_tokens_byte += num_new_tokens

        num_tokens_backbone = self.req_id_to_hat_info[
            request.request_id].num_computed_tokens_backbone

        num_tokens_need_slot = min(
            self.req_id_to_hat_info[
                request.request_id].num_computed_tokens_byte +
            num_lookahead_tokens, self.max_model_len)

        list_num_tokens_need_slot = []
        list_num_tokens_cache = []
        list_num_computed_tokens = []
        for kv_cache_group_spec in self.kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if isinstance(kv_cache_spec, FullAttentionSpec):
                list_num_tokens_need_slot.append(num_tokens_backbone)
                list_num_tokens_cache.append(num_tokens_backbone)
                list_num_computed_tokens.append(num_tokens_backbone -
                                                new_slots_needed_backbone)
            elif isinstance(kv_cache_spec, SlidingWindowSpec):
                list_num_tokens_need_slot.append(num_tokens_need_slot)
                list_num_tokens_cache.append(num_tokens_need_slot -
                                             num_lookahead_tokens)
                list_num_computed_tokens.append(num_tokens_need_slot -
                                                num_new_tokens -
                                                num_lookahead_tokens)

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               list_num_computed_tokens)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=list_num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
        )

        if num_blocks_to_allocate:
            print("Allocated", num_blocks_to_allocate, "blocks")
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            if copy_state is not None:
                print("Tried", request.request_id, "but failed")
                print("Copying state", copy_state)
                self.req_id_to_hat_info[request.request_id] = copy_state
            else:
                print("Tried", request.request_id, "but failed")
                print("Deleting state")
                del self.req_id_to_hat_info[request.request_id]
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert all(not blocks for blocks in new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, list_num_tokens_need_slot)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        self.coordinator.cache_blocks(
            request, self.req_to_block_hashes[request.request_id],
            list_num_tokens_cache)

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted 
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        if request.request_id in self.req_id_to_hat_info:
            del self.req_id_to_hat_info[request.request_id]
        self.coordinator.free(request.request_id)


class HATKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        caching_hash_fn: Callable,
        enable_kv_cache_events: bool,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len

        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching,
                                    enable_kv_cache_events)

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                caching_hash_fn=caching_hash_fn,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: List[int],
            new_computed_blocks: tuple[list[KVCacheBlock]]) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id, num_tokens[i], new_computed_blocks[i])
        return num_blocks_to_allocate

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: List[int]) -> tuple[list[KVCacheBlock]]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        new_blocks = []
        for i, manager in enumerate(self.single_type_managers):
            new_blocks.append(
                manager.allocate_new_blocks(request_id, num_tokens[i]))
        return new_blocks

    def cache_blocks(self, request: Request, block_hashes: list[BlockHash],
                     num_computed_tokens: List[int]) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            block_hashes: The block hashes of the request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.cache_blocks(request, block_hashes, num_computed_tokens[i])

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: List[int]) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace 
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.remove_skipped_blocks(request_id, num_computed_tokens[i])
