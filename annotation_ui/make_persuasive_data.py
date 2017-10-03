#!/usr/bin/env python
import json
import pandas as pd
import toolz
todos = json.load(open('../gruntwork/persuade_0_persuasive_anno_todo.json'))
BATCH_SIZE=6
print("Total:", len(todos))
batches = list(toolz.partition_all(BATCH_SIZE, todos))
print(pd.Series([len(batch) for batch in batches]).value_counts())
if len(batches[-1]) != BATCH_SIZE:
    print("Tacking on extra to the last batch.")
    batches[-1] = (batches[-1] + batches[0])[:BATCH_SIZE]
assert len(batches[-1]) == BATCH_SIZE
pd.DataFrame(dict(task=[json.dumps(batch) for batch in batches])).to_csv('persuasive-task.csv', index=False)
