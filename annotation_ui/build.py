#!/usr/bin/env python

from jinja2 import Template
import pandas as pd

def build_hit(basename):
    example_task = pd.read_csv(f'{basename}-task.csv').task.iloc[0]
    t = Template(open(f'{basename}.html').read())
    open(f'{basename}-dev.html', 'w').write(t.render(dev=True, task=example_task))
    open(f'{basename}-prod.html', 'w').write(t.render(dev=False))

build_hit('anno')
build_hit('compare-anno')
build_hit('persuasive')
