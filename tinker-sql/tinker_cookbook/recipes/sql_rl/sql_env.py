import json
import pandas as pd
import asyncio
import chz
from tinker import ModelInput
from tinker_cookbook.completers import (
    StopCondition,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, ToolCall, ContentPart
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.sql_rl.sql_utils import verify_format_and_extract, execute_sql_wrapper_single
from tinker_cookbook.recipes.sql_rl.grader import grade
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.recipes.sql_rl.prompts import system_prompt, SQL_TOOLS
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Literal, cast, Tuple, Any
from functools import partial
import math
import re
import os
import logging
from typing import List
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import wandb

logger = logging.getLogger(__name__)

qwen_one_shot_prompt = [
    Message(role="user", content="Database Engine:\nSQLite\n\nDatabase Schema:\nCREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\nExternal Knowledge:\npig is the species of.\n\nQuestion:\nhow many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.\n<function_call>{\"name\": \"sql\", \"args\": {\"query\": \"SELECT COUNT(*) FROM animals WHERE species = 'pig';\"}}</function_call>"),
    Message(role="user", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

general_one_shot_prompt = [
    Message(role="user", content="Database Engine:\nSQLite\n\nDatabase Schema:\nCREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\nExternal Knowledge:\npig is the species of.\n\nQuestion:\nhow many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.\n<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>"),
    Message(role="user", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

kimi_one_shot_prompt = [
    Message(role="user", content="Database Engine:\nSQLite\n\nDatabase Schema:\nCREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\nExternal Knowledge:\npig is the species of.\n\nQuestion:\nhow many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.", tool_calls=[ToolCall(function=ToolCall.FunctionBody(name="execute_sql_query", arguments='{"query": "SELECT COUNT(*) FROM animals WHERE species = \'pig\';"}'), id="functions.execute_sql_query:0")]),
    Message(role="tool", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n", tool_call_id="functions.execute_sql_query:0", name="execute_sql_query"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

class SQLEnv(Env):
    def __init__(self, question_id: str, question: str, gold_answer: int, grading_method: str, renderer: Renderer, db_file, timeout, db_modification_script: str | None, dump_path: str | None = None, use_convo_prefix: bool = True, max_output_tokens_per_turn: int = 3072,
    max_input_tokens: int = 32768, model_name: str = "qwen", max_turns: int = 5, noise_rate: float = 0.0):

        self.renderer: Renderer = renderer
        self.noise_rate = noise_rate
        self.turns: list[Message] = []
        self.gold_answer: int = gold_answer
        self.db_file = db_file
        self.timeout = timeout
        self.num_turn = 0
        self.question = question
        self.question_id = question_id
        self.db_modification_script = db_modification_script
        self.dump_path = dump_path
        self.grading_method = grading_method
        self.use_convo_prefix = use_convo_prefix
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_input_tokens = max_input_tokens
        self.model_name = model_name
        self.max_turns = max_turns

        if "Qwen3" in self.model_name:
            self.convo_prefix = qwen_one_shot_prompt
        elif "Kimi" in self.model_name:
            self.convo_prefix = kimi_one_shot_prompt
        else:
            self.convo_prefix = general_one_shot_prompt

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @property
    def _obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""

        if self.use_convo_prefix:
            convo = self.renderer.create_conversation_prefix_with_tools(tools=SQL_TOOLS, system_prompt=system_prompt) + self.convo_prefix + [Message(role="user", content=self.question)] + self.turns
        elif not self.use_convo_prefix:
            convo = self.renderer.create_conversation_prefix_with_tools(tools=SQL_TOOLS, system_prompt=system_prompt) + [Message(role="user", content=self.question)] + self.turns

        return self.renderer.build_generation_prompt(convo)

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._obs, self.stop_condition

    def _parse_tool_calls(self, tool_calls: List[ToolCall] = []) -> Tuple[List[str|None], List[str|None]]:
        sqls = []
        errors = []
        for tool_call in tool_calls:
            arguments = tool_call.function.arguments
            try:
                sql = json.loads(arguments)
                next_sql = sql["query"]
                next_error = None
            except Exception as e:
                next_sql = None
                next_error = f"Error in parsing tool call arguments: {e}"
            sqls.append(next_sql)
            errors.append(next_error)
        return sqls, errors
    
    def _get_last_text_part(self, content: str | list[ContentPart]) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            for part in reversed(content):
                if part["type"] == "text":
                    return part["text"]
        return ""

    async def _get_user_turn(self, action_text: str, tool_calls: List[ToolCall] = []) -> tuple[List[Message]|None, float]:
        # check if there is a solution
        if not action_text.endswith('</solution>'):
            # this means this turn is an intermediate step involving tool calls
            messages = []
            sqls, errors = self._parse_tool_calls(tool_calls)
            index = 0
            for sql, error in zip(sqls, errors):
                if sql is None and error is None:
                    assert False, "Should not reach here"
                elif sql is None and error is not None:
                    messages.append(Message(role="tool", content=f"Your previous action is invalid due to error: {error}. Follow the format of outputting a sql tool call.", tool_call_id=tool_calls[index].id, name="execute_sql_query"))
                    index += 1
                    continue

                assert sql is not None and error is None

                sql_output = await execute_sql_wrapper_single(self.db_file, sql, self.timeout)
                pred_results, error = sql_output
                

                if pred_results is None:
                    messages.append(Message(role="tool", content=error, tool_call_id=tool_calls[index].id, name="execute_sql_query"))
                else:
                    df = pd.DataFrame(pred_results)
                    res = df.to_string(index=False)
                    if len(df) > 50:
                        # just truncate
                        truncated_df = df.head(50)
                        res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                            index=False
                        )  # or index=True if you want row numbers
                    else:
                        res = "SQL execution results: " + res
                    # print(f"SQL output: {res}")
                    response = f"{res}\nYou have {self.max_turns - self.num_turn} turns left to complete the task."
                    
                    messages.append(Message(role="tool", content=response, tool_call_id=tool_calls[index].id, name="execute_sql_query"))

                index += 1

            if len(messages) == 0:
                messages.append(Message(role="user", content="Your previous action is invalid. Follow the format of outputting a sql tool call or a final solution."))
            return messages, 0.0
        else:
            is_valid, _, pred_sql, _ = verify_format_and_extract(action_text)

            if not is_valid:
                return [Message(role="user", content="Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again.")], -1.0

            if self.dump_path is not None:
                with open(os.path.join(self.dump_path, f"{self.question_id}_pred.sql"), "w") as f:
                    f.write(pred_sql)
                wandb.termlog(f"Prediction for Problem {self.question_id}: {pred_sql}")
            pred = await execute_sql_wrapper_single(self.db_file, pred_sql, self.timeout)
            ref = await execute_sql_wrapper_single(self.db_file, self.gold_answer, self.timeout)

            pred_results, error = pred
            gt_results, gt_error = ref

            if pred_results is None:
                return None, 0.0
            elif gt_results is None:
                return None, 0.0
            else:
                if grade(gt_results, pred_results, self.grading_method)[0]:
                    return None, 1.0
                else:
                    return None, 0.0


    def _is_done(self, action: str) -> bool:
        if self.num_turn == self.max_turns:
            return True
        return "<solution>" in action and "</solution>" in action


    async def step(self, action: Action) -> StepResult:
        self.num_turn += 1

        # step 1: parse the action tokens into a message
        # this step is specific to our library, but usually templated, so you can just copy it.
        (action_message, _parse_success) = self.renderer.parse_response(action)

        # step 2: based on the string answer, we compute the reward and the user turn.
        user_turn, reward = await self._get_user_turn(
            self._get_last_text_part(action_message["content"]),
            action_message["tool_calls"] if "tool_calls" in action_message else []
        )

        # step 3: update the conversation history
        self.turns.append({"role": "assistant", "content": action_message["content"]})
        if "tool_calls" in action_message and action_message["tool_calls"] != []:
            self.turns[-1]["tool_calls"] = action_message["tool_calls"]
    
        # print("=" * 100)
        # print(f"Action: {self._get_last_text_part(action_message['content'])}")
        # if "tool_calls" in action_message:
        #     for tool_call in action_message["tool_calls"]:
        #         print(f"Tool Call (id={tool_call.id}): {tool_call.function.name} with arguments {tool_call.function.arguments}")

        if user_turn is not None:
            self.turns += user_turn
            # print(f"Next user turn:")
            # for user_turn_1 in user_turn:
            #     if user_turn_1['role'] == 'tool':
            #         print(f"Tool Call ID: {user_turn_1.get('tool_call_id')}")
            #     print(f"Role: {user_turn_1['role']}, Content: {user_turn_1['content']}")
        episode_done = self._is_done(action_message['content'])

        if self._obs.length + self.max_output_tokens_per_turn > self.max_input_tokens:
            episode_done = True
            print("Observation too long, marking episode as done.")
        
        # if episode_done:
        #     print(f"Reward: {reward}")
        #     print("=" * 100)

        # apply PGFC reward correction if noise_rate is set
        if self.noise_rate > 0.0 and episode_done:
            reward = self.noise_rate if reward > 0 else self.noise_rate - 1.0

        # step 4: return the step result
        step_result = StepResult(
            next_observation=self._obs,
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward= 0 if not episode_done else reward
        )

        return step_result

class BIRDDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data_path: str,
        db_path: str|List[str]|None,
        db_modification_script_path: str,
        timeout: int,
        model_name: str,
        split: Literal["train", "test"] = "train",
        n_epochs: int = 1,
        num_data: int = -1,
        use_convo_prefix: bool = True,
        max_output_tokens_per_turn: int = 3072,
        max_input_tokens: int = 32768,
        max_turns: int = 5,
        dump_path: str | None = None,
        noise_rate: float = 0.0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if isinstance(data_path, list):
            # curriculum learning setup
            stage_1_data = cast(Dataset, load_dataset("parquet", data_files=data_path[0], keep_in_memory=True)["train"])
            stage_2_data = cast(Dataset, load_dataset("parquet", data_files=data_path[1], keep_in_memory=True)["train"])
            stage_3_data = cast(Dataset, load_dataset("parquet", data_files=data_path[2], keep_in_memory=True)["train"])
            stage_4_data = cast(Dataset, load_dataset("parquet", data_files=data_path[3], keep_in_memory=True)["train"])
            # stage 1: 2 epochs
            stage_1_data = concatenate_datasets([stage_1_data for _ in range(2)])
            print(f"Stage 1 data length after 2 epochs: {len(stage_1_data)}")
            # stage 2: 10 epochs
            stage_2_data = concatenate_datasets([stage_2_data for _ in range(10)])
            print(f"Stage 2 data length after 10 epochs: {len(stage_2_data)}")
            # stage 3: 12 epochs
            stage_3_data = concatenate_datasets([stage_3_data for _ in range(12)])
            print(f"Stage 3 data length after 12 epochs: {len(stage_3_data)}")
            # stage 4: 14 epochs
            stage_4_data = concatenate_datasets([stage_4_data for _ in range(14)])
            print(f"Stage 4 data length after 14 epochs: {len(stage_4_data)}")
            # merge all stages
            self.ds = concatenate_datasets([stage_1_data, stage_2_data, stage_3_data, stage_4_data])
            self.group_sizes = [8 for _ in range(len(stage_1_data))] + [16 for _ in range(len(stage_2_data))] + [32 for _ in range(len(stage_3_data))] + [64 for _ in range(len(stage_4_data))]

        elif isinstance(data_path, str):
            # regular setup
            self.ds = cast(Dataset, load_dataset("parquet", data_files=data_path, keep_in_memory=True)["train"])
            if split == "train":
                self.ds = self.ds.shuffle(seed=0)
                if num_data > 0:
                    self.ds = self.ds.select(range(num_data))
                if n_epochs > 1:
                    self.ds = concatenate_datasets([self.ds for _ in range(n_epochs)])
            if split == "test":
                all_ds = []
                for i in range(n_epochs):
                    ds = deepcopy(self.ds)
                    # append i to data source
                    ds = ds.map(lambda x: {"data_source": f"{x['data_source']}_{i}"})
                    all_ds.append(ds)
                self.ds = concatenate_datasets(all_ds)
            self.group_sizes = [group_size for _ in range(len(self.ds))]
        else:
            raise ValueError("db_path must be a string or a list of strings")
        
        

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.db_path = db_path
        self.timeout = timeout
        self.db_modification_script_path = db_modification_script_path
        self.dump_path = dump_path
        self.use_convo_prefix = use_convo_prefix
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_input_tokens = max_input_tokens
        self.model_name = model_name
        self.max_turns = max_turns
        self.noise_rate = noise_rate

    @classmethod
    def question_suffix(cls) -> str:
        return ""

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        group_sizes = self.group_sizes[batch_start: batch_end]
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row, group_size in zip(self.ds.select(range(batch_start, batch_end)), group_sizes)
            if (builder := self._make_env_group_builder(row, group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def set_dump_path(self, dump_path: str) -> None:
        self.dump_path = dump_path

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem_id = f"{x['data_source']}_{x['question_id']}"
        problem = x["prompt"][1]["content"]
        answer = x["reward_spec"]["ground_truth"]
        grading_method = x["reward_spec"].get("grading_method", "multiset")
        dataset_name = x["data_source"]
        db_id = x["db_id"]
        db_file = f"{self.db_path}/{db_id}/{db_id}.sqlite"
        if os.path.exists(f"{self.db_modification_script_path}/{x['question_id']}.sql"):
            db_modification_script = f"{self.db_modification_script_path}/{x['question_id']}.sql"
        else:
            db_modification_script = None
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                SQLEnv, problem_id, problem, answer, grading_method, self.renderer, db_file, self.timeout, db_modification_script, self.dump_path, self.use_convo_prefix, self.max_output_tokens_per_turn, self.max_input_tokens, self.model_name, self.max_turns, self.noise_rate
            ),
            num_envs=group_size,
            dataset_name=dataset_name,
        )


@chz.chz
class BIRDDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    renderer_name: str
    train_group_size: int
    base_url: str | None = None
    model_name: str
    train_data_path: str | None
    test_data_path: str | None
    db_modification_script_path: str | None
    db_path: str | None
    timeout: int = 60
    add_noise: str | None = None
    n_epochs: int = 1
    num_data: int = -1
    use_convo_prefix: bool = True
    max_output_tokens_per_turn: int = 3072
    max_input_tokens: int = 32768
    max_turns: int = 5
    curriculum_learning: bool = False
    noise_rate: float = 0.0

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        sql_renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        if self.curriculum_learning:
            train_data_path = [
                f"{self.train_data_path}/stage_1.parquet",
                f"{self.train_data_path}/stage_2.parquet",
                f"{self.train_data_path}/stage_3.parquet",
                f"{self.train_data_path}/stage_4.parquet",
            ]
        else:
            train_data_path = self.train_data_path
        if "db" in self.add_noise:
            assert "/databases" in self.db_path, "db_path must contain /databases if db noise is used"
            db_path = self.db_path.replace("/databases", "/original_databases")
        else:
            db_path = self.db_path

        test_data_path = self.test_data_path

        # log information about datasets
        logger.info(f"Training data path: {train_data_path}")
        logger.info(f"Test data path: {test_data_path}")
        logger.info(f"Database path: {db_path}")

        training_dataset = BIRDDataset(
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            renderer=sql_renderer,
            data_path=train_data_path,
            db_modification_script_path=self.db_modification_script_path,
            timeout=self.timeout,
            db_path=db_path,
            split="train",
            n_epochs=self.n_epochs,
            num_data=self.num_data,
            use_convo_prefix=self.use_convo_prefix,
            max_output_tokens_per_turn=self.max_output_tokens_per_turn,
            max_input_tokens=self.max_input_tokens,
            max_turns=self.max_turns,
            model_name=self.model_name,
            noise_rate=self.noise_rate,
        )
        test_dataset = BIRDDataset(
            batch_size=self.batch_size,
            group_size=1,
            renderer=sql_renderer,
            data_path=test_data_path,
            db_modification_script_path=None,
            timeout=self.timeout,
            db_path=self.db_path,
            split="test",
            n_epochs=1,
            use_convo_prefix=self.use_convo_prefix,
            max_output_tokens_per_turn=self.max_output_tokens_per_turn,
            max_input_tokens=self.max_input_tokens,
            max_turns=self.max_turns,
            model_name=self.model_name,
            noise_rate=0.0,
        )
        return training_dataset, test_dataset
