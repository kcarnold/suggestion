#!/usr/bin/env python
import json
import pandas as pd
groups = json.load(open('../gruntwork/persuade_0_persuasive_anno_todo.json'))
print("Total:", len(groups))
pd.DataFrame(dict(task=[json.dumps(group) for group in groups])).to_csv('persuasive-task.csv', index=False)
