# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate responses given a dataset of chat-format prompts.

Standalone generation notes
---------------------------
1. Parquet/PyArrow may reconstruct nested multimodal chat objects with
   NumPy arrays and union-style null fields. We canonicalize them before
   sending them to the OpenAI-compatible rollout server.

2. PPO rollout configuration normally defaults to:

       actor_rollout_ref.rollout.load_format=dummy

   This is appropriate during RL/OPD training because actor weights are
   synchronized into the rollout worker afterwards.

   main_generation_server is different: it launches a standalone inference
   server and there is no subsequent actor -> rollout weight synchronization.

   Therefore standalone generation MUST NOT use load_format=dummy.
   If dummy is inherited from the PPO config, this script automatically
   changes it to "auto" so that the model weights are actually loaded from
   actor_rollout_ref.model.path.
"""

import asyncio
import json
import os
from pprint import pprint

import aiohttp
import hydra
import numpy as np
import pandas as pd
import ray

from omegaconf import ListConfig, OmegaConf
from openai.types.chat import ChatCompletion

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


DEBUG_GENERATION_SERVER = (
    os.environ.get("VERL_GEN_DEBUG", "0") == "1"
)

_DEBUG_REQUEST_PRINTED = False


# =============================================================================
# JSON / parquet normalization
# =============================================================================

def to_jsonable(obj):
    """
    Recursively convert NumPy/PyArrow reconstructed objects into ordinary
    JSON-safe Python objects.
    """

    if isinstance(obj, np.ndarray):
        return [
            to_jsonable(x)
            for x in obj.tolist()
        ]

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            k: to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            to_jsonable(v)
            for v in obj
        ]

    return obj


def normalize_chat(chat):
    """
    Canonicalize one OpenAI-format conversation loaded from parquet.

    PyArrow may unify heterogeneous structs. For example:

        {
            "type": "text",
            "text": "question",
            "image_url": None,
        }

    or:

        {
            "type": "image_url",
            "image_url": {...},
            "text": None,
        }

    Some multimodal processors inspect field presence rather than only the
    declared "type". Therefore rebuild each block according to its type and
    discard irrelevant null fields.
    """

    chat = to_jsonable(chat)

    if isinstance(chat, str):
        try:
            chat = json.loads(chat)
        except json.JSONDecodeError as e:
            raise TypeError(
                f"Chat is a string but is not valid JSON: {chat[:200]!r}"
            ) from e

    if not isinstance(chat, list):
        raise TypeError(
            f"chat must be list, got {type(chat)}"
        )

    output = []

    for msg_idx, msg in enumerate(chat):

        msg = to_jsonable(msg)

        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except json.JSONDecodeError as e:
                raise TypeError(
                    f"message {msg_idx} must be dict/JSON dict"
                ) from e

        if not isinstance(msg, dict):
            raise TypeError(
                f"message {msg_idx} must be dict, got {type(msg)}"
            )

        role = msg.get("role")

        if role is None:
            raise ValueError(
                f"message {msg_idx} has no role"
            )

        content = to_jsonable(
            msg.get("content", "")
        )

        # Preserve useful message-level fields where possible.
        clean_msg = {
            k: to_jsonable(v)
            for k, v in msg.items()
            if k != "content" and v is not None
        }

        clean_msg["role"] = role

        # ---------------------------------------------------------------------
        # Plain text message
        # ---------------------------------------------------------------------

        if isinstance(content, str):
            clean_msg["content"] = content
            output.append(clean_msg)
            continue

        if not isinstance(content, list):
            raise TypeError(
                f"message {msg_idx} content must be str/list, "
                f"got {type(content)}"
            )

        clean_content = []

        # ---------------------------------------------------------------------
        # Structured multimodal content
        # ---------------------------------------------------------------------

        for part_idx, part in enumerate(content):

            part = to_jsonable(part)

            if isinstance(part, str):

                clean_content.append({
                    "type": "text",
                    "text": part,
                })

                continue

            if not isinstance(part, dict):
                raise TypeError(
                    f"message {msg_idx} content[{part_idx}] must be dict, "
                    f"got {type(part)}"
                )

            part_type = part.get("type")

            # -----------------------------------------------------------------
            # Text
            # -----------------------------------------------------------------

            if part_type in {
                "text",
                "input_text",
            }:

                text = part.get("text")

                if text is None:
                    text = ""

                clean_content.append({
                    "type": "text",
                    "text": str(text),
                })

            # -----------------------------------------------------------------
            # OpenAI image_url
            # -----------------------------------------------------------------

            elif part_type == "image_url":

                image_url = part.get("image_url")

                if isinstance(image_url, dict):

                    url = image_url.get("url")
                    detail = image_url.get("detail")

                elif isinstance(image_url, str):

                    url = image_url
                    detail = None

                else:

                    raise TypeError(
                        "image_url block has invalid image_url field: "
                        f"{type(image_url)}"
                    )

                if not url:
                    raise ValueError(
                        "image_url block contains no URL"
                    )

                clean_image_url = {
                    "url": url
                }

                if detail is not None:
                    clean_image_url["detail"] = detail

                clean_content.append({
                    "type": "image_url",
                    "image_url": clean_image_url,
                })

            # -----------------------------------------------------------------
            # Native image block
            # -----------------------------------------------------------------

            elif part_type in {
                "image",
                "input_image",
            }:

                image_ref = part.get("image")

                if image_ref is None:

                    image_url = part.get("image_url")

                    if isinstance(image_url, dict):
                        image_ref = image_url.get("url")

                    elif isinstance(image_url, str):
                        image_ref = image_url

                # Some processors use `url` directly.
                if image_ref is None:
                    image_ref = part.get("url")

                if image_ref is None:
                    raise ValueError(
                        "image block contains no image reference"
                    )

                clean_content.append({
                    "type": "image",
                    "image": image_ref,
                })

            # -----------------------------------------------------------------
            # Other content types
            # -----------------------------------------------------------------

            else:

                clean_part = {
                    k: to_jsonable(v)
                    for k, v in part.items()
                    if v is not None
                }

                clean_content.append(
                    clean_part
                )

        clean_msg["content"] = clean_content
        output.append(clean_msg)

    return output


# =============================================================================
# Debug helpers
# =============================================================================

def redact_for_debug(obj):
    """
    Redact base64 image payloads when printing debug requests.
    """

    if isinstance(obj, dict):

        out = {}

        for key, value in obj.items():

            if (
                key in {"url", "image"}
                and isinstance(value, str)
                and value.startswith("data:image/")
            ):

                out[key] = (
                    value[:60]
                    + "..."
                    + f"[total_chars={len(value)}]"
                )

            else:

                out[key] = redact_for_debug(value)

        return out

    if isinstance(obj, list):
        return [
            redact_for_debug(x)
            for x in obj
        ]

    return obj


def print_chat_debug(chat, title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print("chat type:", type(chat))
    print("num messages:", len(chat))

    print(
        json.dumps(
            redact_for_debug(chat),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 100)
    print()


# =============================================================================
# Python-list splitting
# =============================================================================

def split_contiguous(items, num_chunks):
    """
    Split a Python list into contiguous chunks.

    Do not convert nested chat structures into np.array(), because NumPy may
    infer extra dimensions from regular nested message structures.
    """

    if num_chunks <= 0:
        raise ValueError(
            f"num_chunks must be > 0, got {num_chunks}"
        )

    n = len(items)

    base, remainder = divmod(
        n,
        num_chunks,
    )

    chunks = []

    start = 0

    for i in range(num_chunks):

        size = (
            base
            + (1 if i < remainder else 0)
        )

        end = start + size

        chunks.append(
            items[start:end]
        )

        start = end

    assert sum(
        len(chunk)
        for chunk in chunks
    ) == n

    return chunks


# =============================================================================
# Standalone load-format safeguard
# =============================================================================

def ensure_standalone_load_format(config):
    """
    main_generation_server launches a standalone rollout server.

    PPO/OPD training typically uses load_format=dummy because the actor later
    synchronizes its weights into vLLM. That synchronization does NOT happen in
    standalone generation.

    Therefore, if the inherited rollout config says "dummy", replace it with
    "auto" so vLLM actually loads the model from model.path.
    """

    rollout_config = config.actor_rollout_ref.rollout

    load_format = rollout_config.get(
        "load_format",
        None,
    )

    if load_format == "dummy":

        print()
        print("=" * 100)
        print("STANDALONE LOAD FORMAT SAFEGUARD")
        print("=" * 100)

        print(
            "Detected actor_rollout_ref.rollout.load_format=dummy."
        )

        print(
            "main_generation_server has no actor -> rollout weight "
            "synchronization step."
        )

        print(
            "Overriding standalone rollout load_format:"
        )

        print(
            "    dummy -> auto"
        )

        print(
            "Model weights will be loaded from:"
        )

        print(
            f"    {config.actor_rollout_ref.model.path}"
        )

        print("=" * 100)
        print()

        rollout_config.load_format = "auto"

    else:

        print()
        print(
            "Standalone rollout load_format:",
            load_format,
        )
        print()


# =============================================================================
# Start rollout servers
# =============================================================================

async def start_server(config):

    tp_size = (
        config
        .actor_rollout_ref
        .rollout
        .tensor_model_parallel_size
    )

    num_replicas = (
        config.trainer.n_gpus_per_node
        * config.trainer.nnodes
    ) // tp_size

    rollout_config = (
        config.actor_rollout_ref.rollout
    )

    model_config = (
        config.actor_rollout_ref.model
    )

    rollout_server_class = (
        get_rollout_replica_class(
            config.actor_rollout_ref.rollout.name
        )
    )

    rollout_servers = [

        rollout_server_class(

            replica_rank=replica_rank,

            config=rollout_config,

            model_config=model_config,

            gpus_per_node=config.trainer.n_gpus_per_node,

        )

        for replica_rank in range(
            num_replicas
        )
    ]

    await asyncio.gather(
        *[
            server.init_standalone()
            for server in rollout_servers
        ]
    )

    server_handles = [
        server._server_handle
        for server in rollout_servers
    ]

    server_addresses = [
        server._server_address
        for server in rollout_servers
    ]

    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    print()
    print(
        "Standalone rollout servers:",
        server_addresses,
    )
    print()

    return (
        server_handles,
        server_addresses,
    )


# =============================================================================
# HTTP request
# =============================================================================

async def submit_request(
    server_address,
    **chat_complete_request,
):

    global _DEBUG_REQUEST_PRINTED

    extra_headers = (
        chat_complete_request.pop(
            "extra_headers",
            {},
        )
    )

    # Final JSON-safety pass immediately before serialization.
    chat_complete_request = to_jsonable(
        chat_complete_request
    )

    if (
        DEBUG_GENERATION_SERVER
        and not _DEBUG_REQUEST_PRINTED
    ):

        print()
        print("=" * 100)
        print("FIRST HTTP CHAT COMPLETION REQUEST")
        print("=" * 100)

        print(
            json.dumps(
                redact_for_debug(
                    chat_complete_request
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

        print("=" * 100)
        print()

        _DEBUG_REQUEST_PRINTED = True

    timeout = aiohttp.ClientTimeout(
        total=None
    )

    async with aiohttp.ClientSession(
        timeout=timeout
    ) as session:

        async with session.post(

            url=(
                f"http://{server_address}"
                "/v1/chat/completions"
            ),

            headers={
                "Authorization":
                    "Bearer token-abc123",
                **extra_headers,
            },

            json=chat_complete_request,

        ) as resp:

            response_text = await resp.text()

            if resp.status >= 400:

                raise RuntimeError(
                    "Rollout server returned "
                    f"HTTP {resp.status}:\n"
                    f"{response_text[:5000]}"
                )

            try:

                data = json.loads(
                    response_text
                )

            except json.JSONDecodeError as e:

                raise RuntimeError(
                    "Rollout server returned non-JSON response:\n"
                    f"{response_text[:5000]}"
                ) from e

            return ChatCompletion(
                **data
            )


# =============================================================================
# Generate on one replica
# =============================================================================

async def generate_per_replica(
    server_address,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
):

    requests = []

    for messages in chat_lst:

        # Final normalization before request construction.
        messages = normalize_chat(
            messages
        )

        for _ in range(n_samples):

            requests.append({

                "model": model_path,

                "messages": messages,

                **sampling_params,

            })

    if not requests:
        return []

    tasks = [

        submit_request(
            server_address,
            **req,
        )

        for req in requests
    ]

    return await asyncio.gather(
        *tasks
    )


# =============================================================================
# Generate over replicas
# =============================================================================

async def generate(
    server_addresses: list,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
):

    num_replicas = len(
        server_addresses
    )

    chat_chunks = split_contiguous(
        chat_lst,
        num_replicas,
    )

    assert (
        len(server_addresses)
        == len(chat_chunks)
    )

    results = await asyncio.gather(
        *[
            generate_per_replica(

                server_addresses[i],

                model_path,

                n_samples,

                sampling_params,

                chat_chunks[i],

            )

            for i in range(
                num_replicas
            )
        ]
    )

    return results


# =============================================================================
# Main
# =============================================================================

@hydra.main(
    config_path="config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):

    # Resolve Hydra first.
    OmegaConf.resolve(
        config
    )

    # -------------------------------------------------------------------------
    # CRITICAL standalone fix
    # -------------------------------------------------------------------------

    ensure_standalone_load_format(
        config
    )

    # Print the EFFECTIVE config after dummy -> auto safeguard.
    pprint(
        OmegaConf.to_container(
            config,
            resolve=True,
        )
    )

    # -------------------------------------------------------------------------
    # Ray
    # -------------------------------------------------------------------------

    ray.init(

        runtime_env={

            "env_vars": {

                "TOKENIZERS_PARALLELISM":
                    "true",

                "NCCL_DEBUG":
                    "WARN",

                "VLLM_USE_V1":
                    "1",

            }

        }

    )

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    n_samples = (
        config
        .actor_rollout_ref
        .rollout
        .n
    )

    temperature = (
        config
        .actor_rollout_ref
        .rollout
        .temperature
    )

    if temperature == 0.0:

        assert (
            n_samples == 1
        ), (
            "When temperature=0, "
            "n_samples must be 1."
        )

    assert (
        n_samples >= 1
    ), (
        "n_samples should always >= 1"
    )

    sampling_params = {

        "temperature":
            temperature,

        "top_p":
            config
            .actor_rollout_ref
            .rollout
            .top_p,

        "max_tokens":
            config
            .actor_rollout_ref
            .rollout
            .response_length,

    }

    print()
    print("=" * 100)
    print("SAMPLING PARAMS")
    print("=" * 100)
    print(sampling_params)
    print("=" * 100)
    print()

    # -------------------------------------------------------------------------
    # Input files
    # -------------------------------------------------------------------------

    train_files = (
        config.data.train_files
    )

    if not isinstance(
        train_files,
        (list, ListConfig),
    ):

        train_files = [
            train_files
        ]

    datasets = []

    for train_file in train_files:

        print(
            "Reading dataset:",
            train_file,
        )

        datasets.append(
            pd.read_parquet(
                train_file
            )
        )

    dataset = pd.concat(
        datasets,
        axis=0,
        ignore_index=True,
    )

    print()
    print(
        "Dataset rows:",
        len(dataset),
    )

    print(
        "Dataset columns:",
        dataset.columns.tolist(),
    )

    # -------------------------------------------------------------------------
    # Normalize chats
    # -------------------------------------------------------------------------

    raw_chat_lst = (
        dataset[
            config.data.prompt_key
        ]
        .tolist()
    )

    chat_lst = [

        normalize_chat(chat)

        for chat in raw_chat_lst

    ]

    print()
    print(
        "Normalized chats:",
        len(chat_lst),
    )

    if (
        DEBUG_GENERATION_SERVER
        and len(chat_lst) > 0
    ):

        print_chat_debug(
            chat_lst[0],
            "FIRST NORMALIZED CHAT",
        )

    # -------------------------------------------------------------------------
    # Start standalone rollout server
    # -------------------------------------------------------------------------

    server_handles, server_addresses = (
        asyncio.run(
            start_server(
                config
            )
        )
    )

    # Prevent handles from being dropped while the HTTP servers are in use.
    assert server_handles

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    gen_results = asyncio.run(

        generate(

            server_addresses,

            config
            .actor_rollout_ref
            .model
            .path,

            n_samples,

            sampling_params,

            chat_lst,

        )

    )

    # -------------------------------------------------------------------------
    # Flatten results
    # -------------------------------------------------------------------------

    flat_results = [

        result

        for replica_results in gen_results

        for result in replica_results

    ]

    expected = (
        len(chat_lst)
        * n_samples
    )

    if len(flat_results) != expected:

        raise RuntimeError(
            "Unexpected generation count: "
            f"got {len(flat_results)}, "
            f"expected {expected}"
        )

    contents = [

        result
        .choices[0]
        .message
        .content

        for result in flat_results

    ]

    # Restore n samples per original prompt.
    results = [

        contents[
            i * n_samples:
            (i + 1) * n_samples
        ]

        for i in range(
            len(chat_lst)
        )

    ]

    assert (
        len(results)
        == len(chat_lst)
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    dataset["responses"] = (
        results
    )

    output_dir = os.path.dirname(
        config.data.output_path
    )

    makedirs(
        output_dir,
        exist_ok=True,
    )

    print()
    print(
        f"Saving results to "
        f"{config.data.output_path}"
    )

    dataset.to_parquet(
        config.data.output_path
    )

    print()
    print("Generation finished.")


if __name__ == "__main__":
    main()