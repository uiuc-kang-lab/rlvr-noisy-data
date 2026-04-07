from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Tuple, Any
from skyrl_gym.tools import SQLCodeExecutorToolGroup
import re
from skyrl_gym.envs.sql.utils import compute_score_single
import os
from typing import Dict
from omegaconf import DictConfig


class SQLEnv(BaseTextEnv):
    """
    Environment for one SQL execution task.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Initialize the environment
        assert "db_id" in extras, "db_id field is required"
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "data" in extras, "data field is required"
        assert "question_id" in extras, "question_id field is required"

        self.db_path = env_config.db_path
        self.db_id = extras["db_id"]
        self.gold_sql = extras["reward_spec"]["ground_truth"]
        self.task = extras["data"]
        self.question_id = extras["question_id"]
        self.grading_method = extras["reward_spec"]["method"]
        self.random_perturb = env_config.random_perturb
        self.db_modification_script_path = env_config["db_modification_scripts_path"]
        if os.path.exists(f"{self.db_modification_script_path}/{self.question_id}.sql"):
            self.db_modification_script = f"{self.db_modification_script_path}/{self.question_id}.sql"
        else:
            self.db_modification_script = None

        if self.task == "synsql":
            self.db_path = os.path.join(
                self.db_path,
                "SynSQL-2.5M/databases",
            )
        elif self.task == "spider":
            self.db_path = os.path.join(
                self.db_path,
                "spider/database",
            )
        elif self.task == "bird-platinum":
            # self.db_path = os.path.join(
                # self.db_path,
                # "bird/train/train_databases",

            # )
            self.db_path = self.db_path
        elif self.task == "bird-dev":
            self.db_path = os.path.join(
                self.db_path,
                "dev/dev_databases",
            )
        elif self.task == "bird-sanity-check":
            self.db_path = self.db_path
        else:
            raise NotImplementedError

        self.db_file = os.path.join(self.db_path, self.db_id, self.db_id + ".sqlite")
        # Check for DB file existence
        if not os.path.exists(self.db_file):
            raise FileNotFoundError(f"Database file not found at: {self.db_file}")

        # override parent
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5

        # Initialize the tools
        self.tool_group = SQLCodeExecutorToolGroup(db_file_path=self.db_path)
        self.init_tool_groups([self.tool_group])

        # Chat history
        # Dict[str, str]: role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> Tuple[str, str, Any]:
        """
        Parse action string to return tool name and corresponding arguments.

        Expected: <sql>...</sql>
        """
        match = re.search(r"<sql>(.*?)</sql>", action, re.DOTALL)
        tool_input = match.group(1) if match else None
        # NOTE: hard code
        # NOTE (shu): in the future imagine can use different tools here
        # Format <tool>tool_name</tool><input>tool_input</input>
        tool_group_name = self.tool_group.get_name()
        tool_name = self.tool_group.get_tool_names()[0]
        return tool_group_name, tool_name, (self.db_id, tool_input, self.max_turns - self.turns, self.db_modification_script)

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score_single(chat_history_str, self.gold_sql, self.db_file, db_modification_script=self.db_modification_script, grading_method=self.grading_method)
        else:
            # No reward for intermediate steps for SQL tasks
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<solution>" in action and "</solution>" in action

    def _validate_action(self, action: str):
        stop_tags = ["</sql>", "</solution>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            tool_group_name, tool_name, tool_input = self._parse_action(action)
            observation = self._execute_tool(tool_group_name, tool_name, tool_input)
        except Exception as e:
            error = str(e)
            observation = None
            tool_group_name = None
            tool_name = None
            tool_input = ""

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {"tool_group": tool_group_name, "tool_name": tool_name, "tool_input": tool_input}
        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
