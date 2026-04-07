"""
SynSQL reward calculation.

- Format reward: <think>...</think> <solution>...</solution>
- Outcome reward: check ground truth and predicted patch similarity
"""

import asyncio
import re
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ProcessPoolExecutor, TimeoutError, ThreadPoolExecutor
import sys, os
import random
from tinker_cookbook.recipes.sql_rl.grader import grade
from copy import deepcopy
import time
import logging
from functools import partial

THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"

MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY_PER_RUN', 16))
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
sem = asyncio.Semaphore(MAX_CONCURRENCY)

# NOTE: bring back reward
def verify_format_and_extract(output: str):
    # if output.count(SOLUTION_START) != 1:
    #     return False, None, None, None
    tail = output.split(SOLUTION_START)[-1]

    if tail.count(SOLUTION_END) != 1:
        return False, None, None, None

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None, None, None

    # thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    # if not thoughts:
    #     return False, None, None, None

    # for m in re.finditer(r"</observation>", pre_solution, re.I):
    #     rest = pre_solution[m.end() :].lstrip()
    #     if not rest.lower().startswith(THINK_START):
    #         return False, None, None, None

    return True, None, solution_text.strip(), None

def execute_sql_single(sql, db_file, exec_timeout):
    conn = sqlite3.connect(
        db_file,
        timeout=5,
        uri=False,
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA query_only=ON;")
        start = time.monotonic()
        
        def progress():
            return 1 if (time.monotonic() - start) > exec_timeout else 0

        conn.set_progress_handler(progress, 10_000)
        cur = conn.execute(sql)
        return cur.fetchall(), None
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            print(f"SQL query timeout after {exec_timeout}s: {sql}")
            return None, f"SQL query timeout after {exec_timeout}s: {sql}"
        return None, f"Error executing SQL: {e}, SQL: {sql}"
    except Exception as e:
        return None, f"Error executing SQL: {e}, SQL: {sql}"
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()

async def execute_sql_wrapper_single(db_file, sql, timeout):
    async with sem:
        fn = partial(execute_sql_single, sql, db_file, exec_timeout=timeout)
        return await asyncio.get_running_loop().run_in_executor(executor, fn)

    return res


async def calculate_reward_single(completion, reference, db_file, timeout=60):
    reward = 0.0
    num_comparisons = 0

    is_valid, _, pred_sql, _ = verify_format_and_extract(completion)
    if not is_valid:
        reward = -1.0
        return reward
    else:
        num_comparisons += 1

    pred = await execute_sql_wrapper_single(db_file, pred_sql, timeout)
    ref = await execute_sql_wrapper_single(db_file, reference, timeout)

    pred_results, _ = pred
    gt_results, _ = ref

    if pred_results is not None and gt_results is not None:
        is_correct, info = grade(list(gt_results), list(pred_results), grading_method="multiset")
        if not is_correct:
            reward = 0.0
        if is_correct:
            reward = 1.0
    else:
        reward = 0.0
    return reward


def compute_score_single(completion, reference, db_file, random_perturb=False):
    try:
        res = calculate_reward_single(completion, reference, db_file, random_perturb=random_perturb)
        return res
    except Exception as e:
        logging.info(f"Unexpected error: {e}; Setting reward as 0")
        return 0
