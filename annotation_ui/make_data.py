#!/usr/bin/env python
import json
import pandas as pd
import toolz
todos = json.load(open('../gruntwork/sent4_1_annotations_todo.json'))
BATCH_SIZE=8
batches = list(toolz.partition_all(BATCH_SIZE, todos))
print(pd.Series([len(batch) for batch in batches]).value_counts())
pd.DataFrame(dict(task=[json.dumps(batch) for batch in batches])).to_csv('task.csv', index=False)
