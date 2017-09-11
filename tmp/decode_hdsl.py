import subprocess
import glob
import os
import json
import dateutil.parser
import datetime
import re

if False:
    pwd = os.getcwd()
    try:
        os.chdir('logs')
        hdsl_files_raw = subprocess.check_output(['grep', '--files-with-matches', '-E', '&p=\d+'] + glob.glob('*.jsonl'))
        hdsl_files = hdsl_files_raw.decode('utf8').split()
        pids = [subprocess.check_output(['grep', '-E', '--only-matching', r'&p=\d+', file]).decode('utf8').strip()[3:] for file in hdsl_files]
        # print('\n'.join(map(','.join, [(sid, max((os.path.getsize(f'logs/{pid}.jsonl'), pid) for sid, pid in pids)[1]) for sid, pids in groups.items()])))
    finally:
        os.chdir(pwd)

def get_log_data(log_file, earliest):
    size = os.path.getsize(log_file),
    meta = None
    num_nexts = 0
    with open(log_file) as f:
        for idx, line in enumerate(f):
            if idx > 50 and meta is None:
                return
            line = json.loads(line)
            if line.get('type') == 'externalAction':
                timestamp = dateutil.parser.parse(line['timestamp'])
                if timestamp < earliest:
                    return
                match = re.match(r'c=(\w+)&p=(\d+)', line['externalAction'])
                if not match:
                    continue
                config, pid = match.groups()
                meta = dict(timestamp=timestamp, config=config, pid=int(pid), participant_id=line['participant_id'], size=size)
            elif line.get('type') == 'next':
                num_nexts += 1
    if meta:
        return dict(meta, num_nexts=num_nexts)


earliest = datetime.datetime(2017, 9, 1)
log_files = []
for log_file in glob.glob('logs/*.jsonl'):
    data = get_log_data(log_file, earliest)
    if data is not None:
        print(data)
        log_files.append(data)

import toolz
participants = []
for pid, group in toolz.groupby('pid', log_files).items():
    participants.append(max(group, key=lambda e: e['size']))

# participants.sort(key=lambda x: x['timestamp'])
participants.sort(key=lambda x: x['pid'])
print('\n'.join('{pid},{participant_id},{num_nexts}'.format(**participant) for participant in participants if participant['num_nexts'] == 12))
    # print(pid,
    #     max(group, key=lambda e: e['timestamp'])['participant_id'],
    #     max(group, key=lambda e: e['size'])['participant_id'],
    #     )
    # print()['participant_id'])