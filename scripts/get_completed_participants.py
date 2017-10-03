import glob
import os
import json
import dateutil.parser
import datetime
import toolz

COMPLETE_NUM_ACTIONS={
    'persuade': 15,
    'synonyms': 16,
    }

TECHNICAL_DIFFICULTIES = '846ch3 rwq22w'.split()
TECHNICAL_DIFFICULTIES.append('qgwjch') # A test participant
INCOMPLETE_BUT_OK = ''.split()

def get_log_data(log_file, earliest):
    size = os.path.getsize(log_file),
    meta = None
    num_nexts = 0
    with open(log_file) as f:
        for idx, line in enumerate(f):
            if idx > 50 and meta is None:
                return
            line = json.loads(line)
            if line.get('type') == 'next':
                num_nexts += 1
            elif line.get('type') == 'login':
                timestamp = dateutil.parser.parse(line['timestamp'])
                if timestamp < earliest:
                    return
                platform_id = line['platform_id']
                meta = dict(timestamp=timestamp, config=line['config'], platform_id=platform_id, participant_id=line['participant_id'], size=size)
    if meta:
        return dict(meta, num_nexts=num_nexts)


def get_logs(earliest):
    log_files = []
    for log_file in glob.glob('logs/*.jsonl'):
        data = get_log_data(log_file, earliest)
        if data is not None:
            print(data)
            log_files.append(data)
    return log_files


earliest = datetime.datetime(2017, 9, 20)
log_files = get_logs(earliest)


# Sona participants may open their link multiple times. Take the one with the largest file.
participants = []
for platform_id, group in toolz.groupby('platform_id', log_files).items():
    if platform_id is None:
        participants.extend(group)
    else:
        participants.append(max(group, key=lambda e: e['size']))

for participant in participants:
    participant['complete'] = (
        participant['num_nexts'] == COMPLETE_NUM_ACTIONS[participant['config']]
        or participant['participant_id'] in INCOMPLETE_BUT_OK)

# For payment:
paid_pids = {f'sona{line.strip()}' for line in open('sona-paid.txt')}

participants.sort(key=lambda x: x['platform_id'] or x['participant_id'])
not_yet_paid = []
for participant in participants:
    platform_id = participant['platform_id']
    if platform_id is None:
        if participant['complete']:
            print(f"Assuming Turk is paid, for {participant['participant_id']}")
    elif platform_id not in paid_pids:
        not_yet_paid.append(participant)

# Dump a CSV by Sona participant id for those we haven't paid who are complete...
print("Complete and not yet paid:")
print('\n'.join(
    '{platform_id},{participant_id}'.format(**participant)
    for participant in not_yet_paid
    if participant['complete']))

print("\nIncomplete and not yet paid:")
print('\n'.join(
    '{platform_id},{participant_id},{num_nexts}'.format(**participant)
    for participant in not_yet_paid
    if not participant['complete']))


# For analysis:
completed_participants = [
    p for p in participants
    if p['participant_id'] not in TECHNICAL_DIFFICULTIES
    and p['complete']]



# Dump a list of participant_ids
for config, group in toolz.groupby('config', completed_participants).items():
    print()
    group = sorted(group, key=lambda x: x['timestamp'])
    print(f'{len(group)} completed in {config}')
    print(f'{config}:',  ' '.join(participant['participant_id'] for participant in group))

