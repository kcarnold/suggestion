#!/usr/bin/env python
import json
import pandas as pd
import toolz
todos = json.load(open('../gruntwork/comparisons_existing_reviews.json'))

import random
random.Random(0).shuffle(todos)

BATCH_SIZE=20
print("Total:", len(todos))
batches = list(toolz.partition_all(BATCH_SIZE, todos))
print(pd.Series([len(batch) for batch in batches]).value_counts())
pd.DataFrame(dict(task=[json.dumps(batch) for batch in batches])).to_csv('compare-anno-task.csv', index=False)
