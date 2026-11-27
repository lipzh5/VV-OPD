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

"""Generate responses given a dataset of prompts."""

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


def to_jsonable(obj):
    """Recursively convert NumPy objects to JSON-safe Python objects."""
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def normalize_chat(chat):
    """Normalize parquet-loaded chat into OpenAI-compatible Python objects."""
    chat = to_jsonable(chat)

    if not isinstance(chat, list):
        raise TypeError(f"chat must be list, got {type(chat)}")

    output = []

    for msg in chat:
        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except json.JSONDecodeError as e:
                raise TypeError(f"message must be dict/JSON dict: {msg[:100]!r}") from e

        if not isinstance(msg, dict):
            raise TypeError(f"message must be dict, got {type(msg)}")

        content = msg.get("content")

        if isinstance(content, str):
            output.append(msg)
            continue

        if not isinstance(content, list):
            raise TypeError(f"content must be list or str, got {type(content)}")

        new_content = []

        for part in content:
            if isinstance(part, str):
                try:
                    part = json.loads(part)
                except json.JSONDecodeError:
                    part = {"type": "text", "text": part}

            if not isinstance(part, dict):
                raise TypeError(f"content part must be dict, got {type(part)}")

            new_content.append(part)

        msg["content"] = new_content
        output.append(msg)

    return output


async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size

    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    rollout_server_class = get_rollout_replica_class(
        config.actor_rollout_ref.rollout.name
    )

    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]

    await asyncio.gather(
        *[server.init_standalone() for server in rollout_servers]
    )

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]

    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    session = None

    try:
        extra_headers = chat_complete_request.pop("extra_headers", {})

        # Critical: make the entire HTTP payload JSON-safe.
        chat_complete_request = to_jsonable(chat_complete_request)

        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)

        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={
                "Authorization": "Bearer token-abc123",
                **extra_headers,
            },
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)

    finally:
        if session is not None:
            await session.close()


async def generate_per_replica(
    server_address,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
):
    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    tasks = [
        submit_request(server_address, **req)
        for req in chat_complete_request
    ]

    return await asyncio.gather(*tasks)


async def generate(
    server_addresses: list,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_numpy: np.ndarray,
):
    num_replicas = len(server_addresses)

    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]

    assert len(server_addresses) == len(chat_sub_array)

    return await asyncio.gather(
        *[
            generate_per_replica(
                server_addresses[i],
                model_path,
                n_samples,
                sampling_params,
                chat_sub_array[i],
            )
            for i in range(num_replicas)
        ]
    )


@hydra.main(
    config_path="config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."

    assert n_samples >= 1, "n_samples should always >= 1"

    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
    }

    train_files = config.data.train_files

    if not isinstance(train_files, (list, ListConfig)):
        train_files = [train_files]

    datasets = []

    for train_file in train_files:
        datasets.append(pd.read_parquet(train_file))

    dataset = pd.concat(
        datasets,
        axis=0,
        ignore_index=True,
    )

    # Parquet/PyArrow may reconstruct nested lists as ndarray.
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [normalize_chat(chat) for chat in chat_lst]

    # Explicit object array avoids NumPy reshaping nested chat structures.
    chat_numpy = np.empty(len(chat_lst), dtype=object)
    chat_numpy[:] = chat_lst

    server_handles, server_addresses = asyncio.run(
        start_server(config)
    )

    gen_results = asyncio.run(
        generate(
            server_addresses,
            config.actor_rollout_ref.model.path,
            n_samples,
            sampling_params,
            chat_numpy,
        )
    )

    import itertools

    results = list(
        itertools.chain.from_iterable(gen_results)
    )

    results = np.array([
        result.choices[0].message.content
        for result in results
    ])

    results = np.reshape(
        results,
        (-1, n_samples),
    )

    assert results.shape == (
        len(chat_lst),
        n_samples,
    )

    results = results.tolist()

    dataset["responses"] = results

    output_dir = os.path.dirname(
        config.data.output_path
    )

    makedirs(
        output_dir,
        exist_ok=True,
    )

    print(
        f"Saving results to {config.data.output_path}"
    )

    dataset.to_parquet(
        config.data.output_path
    )


if __name__ == "__main__":
    main()