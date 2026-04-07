"""
Convert BIRD JSON annotation files to training parquet format.

Reads JSON files from data/ and DB schemas from data/bird_db_schemas.json
to produce parquet files compatible with the tinker-sql training pipeline.

Usage:
  python json_to_parquet.py \
      --input ../../data/bird-corrected-600.json \
      --output ../../tinker-sql/data/bird-plat-588.parquet \
      --data_source clean_train

  python json_to_parquet.py \
      --input ../../data/bird-original-600.json \
      --output ../../tinker-sql/data/noisy-600.parquet \
      --data_source noisy_train

  python json_to_parquet.py \
      --input ../../data/bird-test-600.json \
      --output ../../tinker-sql/data/bird-600-test.parquet \
      --data_source bird_test
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SCHEMA_FILE = os.path.join(DATA_DIR, "bird_db_schemas.json")

SYSTEM_PROMPT = """\
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
{external_knowledge}

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.


Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information.
- You can use SQL tool written within a single <sql>your sql</sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>. Based on this observation, you can think again and refine.
- The returned dataframe will be truncated in 50 rows if observation is too long.
- If you find no further exploration is needed or reaches max turns, you MUST directly provide the final SQL query solution inside <solution>...</solution>.

------------------------ START OF EXAMPLE ------------------------
Question: how many pigs are in the farm?
    Database Schema:
    Table: animals
    - id (INTEGER, PRIMARY KEY)
    - species (TEXT)
    - age (INTEGER)
    - name (TEXT)

<think>I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.</think>
<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>
<observation>
+----------+
| COUNT(*) |
+----------+
|   12     |
+----------+
</observation>
<think>The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ END OF EXAMPLE ------------------------
"""


def convert(input_path: str, output_path: str, data_source: str):
    # Load DB schemas
    with open(SCHEMA_FILE) as f:
        db_schemas = json.load(f)

    with open(input_path) as f:
        data = json.load(f)

    rows = []
    skipped = 0
    for item in data:
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence", "")
        sql = item["SQL"]
        question_id = item["question_id"]
        grading_method = item.get("grading_method", "multiset")

        if not sql or not sql.strip():
            skipped += 1
            continue

        db_details = db_schemas.get(db_id, f"[Database schema for {db_id} not available]")

        user_content = (
            f"{{db_details}}: {db_details}\n"
            f" {{external_knowledge}}: {evidence};\n"
            f" {{question}}: {question}"
        )

        prompt = np.array([
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"content": user_content, "role": "user"},
        ])

        rows.append({
            "data_source": data_source,
            "prompt": prompt,
            "env_class": "text2sql",
            "db_id": db_id,
            "data": "bird-platinum" if "corrected" in input_path or "clean" in data_source else "bird",
            "reward_spec": {"ground_truth": sql, "method": grading_method},
            "extra_info": {"index": len(rows), "split": "dummy"},
            "question_id": question_id,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(rows)} instances to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BIRD JSON to training parquet format")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--data_source", default="clean_train", help="Data source label (clean_train or noisy_train)")
    args = parser.parse_args()

    convert(args.input, args.output, args.data_source)
