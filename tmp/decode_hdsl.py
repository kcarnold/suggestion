import glob
import os
import json
import dateutil.parser
import datetime
import re

COMPLETE_NUM_ACTIONS=18

TECHNICAL_DIFFICULTIES = '7h9r8g p964wg jcqf4w 9qxf5g'.split()
# Also exclude this person, who wrote about things other than restaurants.
TECHNICAL_DIFFICULTIES.append('49g68p')
INCOMPLETE_BUT_OK = 'hfj33r'.split()

def get_log_data(log_file, earliest):
    size = os.path.getsize(log_file),
    meta = None
    num_nexts = 0
    with open(log_file) as f:
        for idx, line in enumerate(f):
            if idx > 50 and meta is None:
                return
            line = json.loads(line)
            if line.get('type') == 'next' or line.get('externalAction') == 'completeSurvey':
                num_nexts += 1
            elif line.get('type') == 'externalAction':
                timestamp = dateutil.parser.parse(line['timestamp'])
                if timestamp < earliest:
                    return
                match = re.match(r'c=(\w+)&p=(\d+)', line['externalAction'])
                if not match:
                    continue
                config, pid = match.groups()
                meta = dict(timestamp=timestamp, config=config, pid=int(pid), participant_id=line['participant_id'], size=size)
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

for participant in participants:
    participant['complete'] = (
        participant['num_nexts'] == COMPLETE_NUM_ACTIONS
        or participant['participant_id'] in INCOMPLETE_BUT_OK)

# For payment:
paid_pids = {int(line.strip()) for line in open('sona-paid.txt')}

participants.sort(key=lambda x: x['pid'])
not_yet_paid = []
for participant in participants:
    if participant['pid'] not in paid_pids:
        not_yet_paid.append(participant)
assert len(not_yet_paid) + len(paid_pids) == len(participants)

# Dump a CSV by Sona participant id for those we haven't paid who are complete...
print("Complete and not yet paid:")
print('\n'.join(
    '{pid},{participant_id}'.format(**participant)
    for participant in not_yet_paid
    if participant['complete']))

print("\nIncomplete and not yet paid:")
print('\n'.join(
    '{pid},{participant_id},{num_nexts}'.format(**participant)
    for participant in not_yet_paid
    if not participant['complete']))


# For analysis:
completed_participants = [
    p for p in participants
    if p['participant_id'] not in TECHNICAL_DIFFICULTIES
    and p['complete']]



# Dump a list of participant_ids
print()
completed_participants.sort(key=lambda x: x['timestamp'])
print(len(completed_participants))
print(' '.join(participant['participant_id'] for participant in completed_participants))

