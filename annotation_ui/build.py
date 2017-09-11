#!/usr/bin/env python

from jinja2 import Template
import pandas as pd

example_task = pd.read_csv('task.csv').task.iloc[0]

t = Template(open('anno.html').read())

open('anno-dev.html', 'w').write(t.render(dev=True, task=example_task))
open('anno-prod.html', 'w').write(t.render(dev=False))

