#!/usr/bin/env python

from jinja2 import Template
import pandas as pd
import hashlib

prod_basepath = 'https://s3.amazonaws.com/megacomplete.net/anno/'

def build_hit(basename):
    for config in ['dev', 'prod']:
        args = {}
        if config == 'dev':
            example_task = pd.read_csv(f'{basename}-task.csv').task.iloc[0]
            t = Template(open(f'{basename}.html').read())
            args['task'] = example_task
            args['dev'] = True
            basepath = ''
        else:
            basepath = prod_basepath
        for resource in ['css', 'js']:
            filename = f'{basename}.{resource}'
            filehash = hashlib.sha1(open(filename, 'rb').read()).hexdigest()
            args[resource] = f'{basepath}{filename}?{filehash}'
        open(f'{basename}-{config}.html', 'w').write(t.render(**args))

# build_hit('anno')
# build_hit('compare-anno')
# build_hit('persuasive')
# build_hit('avoid')
build_hit('pairwise-persuasive')
