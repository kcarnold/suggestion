#!/usr/bin/env python
import json
import pandas as pd
import toolz
todos = json.load(open('../gruntwork/persuade_0_annotations_todo.json'))
BATCH_SIZE=4
print("Total:", len(todos))
batches = list(toolz.partition_all(BATCH_SIZE, todos))
print(pd.Series([len(batch) for batch in batches]).value_counts())
if len(batches[-1]) != BATCH_SIZE:
    print("Tacking on extra to the last batch.")
    batches[-1] = (batches[-1] + batches[0])[:BATCH_SIZE]
assert len(batches[-1]) == BATCH_SIZE
pd.DataFrame(dict(task=[json.dumps(batch) for batch in batches])).to_csv('anno-task.csv', index=False)

