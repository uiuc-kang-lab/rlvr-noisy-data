"""
SynSQL reward calculation.

- Format reward: <think>...</think> <solution>...</solution>
- Outcome reward: check ground truth and predicted patch similarity
"""

import re
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import sys
import random
from copy import deepcopy
from skyrl_gym.envs.sql.grading import grade

THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"


# NOTE: bring back reward
def verify_format_and_extract(output: str):
    if output.count(SOLUTION_START) != 1:
        return False, None, None, None
    pre_solution, tail = output.split(SOLUTION_START, 1)

    if tail.count(SOLUTION_END) != 1:
        return False, None, None, None

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None, None, None

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return False, None, None, None

    for m in re.finditer(r"</observation>", pre_solution, re.I):
        rest = pre_solution[m.end() :].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None, None

    return True, thoughts, solution_text.strip(), None

def remove_null_rows(results):
    return [row for row in results if not all(col is None for col in row)]

def execute_sql_single(db_file, sql, db_modification_script):
    try:
        conn = sqlite3.connect(db_file, autocommit=False)
        cursor = conn.cursor()
        # if db_modification_script:
        #     with open(db_modification_script, "r") as f:
        #         modification_sql = f.read()
        #     try:
        #         cursor.executescript(modification_sql)
        #     except Exception as e:
        #         print(f"WARNING executing DB modification script failed: {e}, db file: {db_file}")
        # conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = deepcopy(cursor.fetchall())
        execution_res = remove_null_rows(execution_res)
        conn.rollback()
        conn.close()
        # print('Successfully executed')
        return db_file, sql, execution_res, 1
    except Exception as e:
        print(f"Error executing SQL: {e}, db file: {db_file}")
        conn.rollback()
        conn.close()
        return db_file, sql, None, 0


def execute_sql_wrapper_single(db_file, sql, timeout, output_str, db_modification_script=None):
    try:
        res = func_timeout(timeout, execute_sql_single, args=(db_file, sql, db_modification_script))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"SQL:\n{sql}\nTime Out!")
        print("-" * 30)
        res = (db_file, sql, None, 0)
    except Exception:
        print(f"Error executing SQL: {e}, db_file: {db_file}")
        res = (db_file, sql, None, 0)

    # Append the output to the tuple
    if isinstance(res, tuple):
        res = res + (output_str,)

    return res


def calculate_reward_single(completion, reference, db_file, timeout=60, db_modification_script=None, grading_method="multiset"):
    reward = 0.0
    num_comparisons = 0

    is_valid, _, pred_sql, _ = verify_format_and_extract(completion)
    if not is_valid:
        reward = -1.0
        return reward
    else:
        num_comparisons += 1

    pred = execute_sql_wrapper_single(db_file, pred_sql, timeout, completion, db_modification_script)
    ref = execute_sql_wrapper_single(db_file, reference, timeout, completion, db_modification_script)

    # print(f"Pred SQL: {pred_sql}")
    # print(f"Ref SQL: {reference}")

    _, _, pred_results, _, _ = pred
    _, _, gt_results, _, _ = ref

    if pred_results is not None and gt_results is not None:
        is_correct, _ = grade(gt_results, pred_results, grading_method=grading_method)
        # print(f"Grading method: {grading_method}")
        # print(f"Grading results: {is_correct}")
        # print(f"Grading message: {message}")


        if is_correct:
            reward = 1.0
        else:
            reward = 0.0
    else:
        reward = 0.0
    return reward


def compute_score_single(completion, reference, db_file, db_modification_script=None, grading_method="multiset"):
    try:
        res = calculate_reward_single(completion, reference, db_file, db_modification_script=db_modification_script, grading_method=grading_method)
        return res
    except Exception as e:
        print(f"Unexpected error: {e}; Setting reward as 0")
        return 0
