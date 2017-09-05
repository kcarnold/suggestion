# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:24:31 2016

@author: kcarnold
"""

from jinja2 import Template

t = Template('''
<!-- Bootstrap v3.0.3 -->
<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />
<section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
<div class="row col-xs-12 col-md-12"><!-- Instructions -->
<p>&nbsp;</p>

<div class="panel panel-primary">
<div class="panel-heading"><strong>Instructions</strong></div>

<div class="panel-body">
<p>Please rate the following restaurant reviews.</p>

<ul>
	<li>Don&#39;t penalize reviews for missing capitalization, numbers, or punctuation. These reviews were written during a study of mobile phone keyboards that lacked some of those keys.</li>
	<li>Use a 1-5 scale, where <strong>1=terrible</strong> and <strong>5=outstanding</strong>.</li>
</ul>
</div>
</div>
<!-- End Instructions --><!-- Content Body -->

<section>

<table border="1"><tbody>
{% for i in range(1, 11) %}<tr><td>{{i}}.</td><td>{{ '${' }}r{{i}}_text{{ '}' }}</td>
<td><table><tbody>
    {% for part in ['Clear', 'Helpful', 'Creative', 'Overall'] %}
    <tr><td>{{part}}</td>
    <td style="white-space: nowrap;">{% for j in range(5) %}<label><input type="radio" name="r{{i}}_{{part}}" value="{{j+1}}"><br>{{j+1}}</label>{% endfor %}</td>
    </tr>
    {% endfor %}
</tbody></table></td></tr>
{% endfor %}
</tbody>
</table>


<p>We&#39;re just developing this HIT, so we&#39;d appreciate your feedback: are the instructions clear? Is the payment fair? Anything else?</p>
<textarea cols="80" name="feedback" placeholder="optional feedback" rows="4"></textarea></section>
<!-- End Content Body --></div>
<!-- close container -->
<style type="text/css">
label {
  text-align: center;
  padding: 0 3px;
}
input[type="radio"] {
  display: inline-block;
}
</style>
</section>

''')

html = t.render()
print(html)
import subprocess
subprocess.Popen('pbcopy', stdin=subprocess.PIPE).communicate(html.encode('utf-8'))
#%%
import json
torate = json.load(open('ui/torate.json'))
import random
random.seed(0)
random.shuffle(torate)
batches = []
for i in range(len(torate) // 10):
    batch = {}
    for j, review in enumerate(torate[10*i:10*(i+1)]):
        review['text'] = review.pop('reviewText')
        for k, v in review.items():
            batch['r{}_{}'.format(j+1, k)] = v
    batches.append(batch)
import pandas as pd
pd.DataFrame(batches).to_csv('rating_batches_of_10.csv', index=False)

#%%
# Now to analyze it.
ratings = []
df = pd.read_csv('/Users/kcarnold/Downloads/Batch_2431258_batch_results-1.csv')
for i, row in df.iterrows():
    for j in range(1, 11):
        v = {}
        for k in row.index:
            for part in ['Answer', 'Input']:
                prefix = '{}.r{}_'.format(part, j)
                if k.startswith(prefix):
                    val = row[k]
                    if part == 'Answer' and not 1 <= val <= 5:
                        continue
                    v[k[len(prefix):]] = val
        ratings.append(v)
df2 = pd.DataFrame(ratings)
df2.to_csv('rating_results_1.csv', index=False)
df2.groupby(['participant_id', 'idx']).mean().to_csv('rating_results_agg.csv')
