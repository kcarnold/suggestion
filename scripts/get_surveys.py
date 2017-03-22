import requests
import zipfile
import json
import io
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

dataCenter = 'harvard.az1'
api_token = os.environ['QUALTRICS_API_TOKEN']
#surveyId = "SV_8HVnUso1f0DZExv"
surveys = dict(
  intro='SV_9GiIgGOn3Snoxwh',
  instructionsQuiz='SV_42ziiSrsZzOdBul',
  postFreewrite='SV_0OCqAQl6o7BiidT',
  postTask='SV_5yztOdf3SX8EtOl',
  postExp='SV_8HVnUso1f0DZExv')

surveyId = "SV_0OCqAQl6o7BiidT"

def get_survey(survey_id):
    return requests.get(
        'http://{}.qualtrics.com/API/v3/surveys/{}'.format(dataCenter, survey_id),
        headers={'x-api-token': api_token}).json()['result']

def download_surveys(out_path):
    for name, survey_id in surveys.items():
        print(name)
        survey = get_survey(survey_id)
        with open(os.path.join(out_path, '{}.qsf'.format(name)), 'w') as f:
            json.dump(survey, f)

def get_responses(surveyId):
    requestCheckProgress = 0
    baseUrl = "https://{0}.qualtrics.com/API/v3/responseexports/".format(dataCenter)
    headers = {
        "content-type": "application/json",
        "x-api-token": api_token,
    }

    # Create the data export.
    downloadRequestResponse = requests.post(baseUrl, json=dict(
        format='json', surveyId=surveyId, useLabels=True), headers=headers)
    progressId = downloadRequestResponse.json()["result"]["id"]
    # print(downloadRequestResponse.text)

    # Report progress until export is ready
    while requestCheckProgress < 100:
      requestCheckUrl = baseUrl + progressId
      requestCheckResponse = requests.get(requestCheckUrl, headers=headers)
      requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
      print("Download is " + str(requestCheckProgress) + " complete")

    # Download and unzip the export file.
    requestDownloadUrl = baseUrl + progressId + '/file'
    requestDownload = requests.get(requestDownloadUrl, headers=headers)
    zf = zipfile.ZipFile(io.BytesIO(requestDownload.content))
    assert len(zf.filelist) == 1
    return json.loads(zf.read(zf.filelist[0]).decode('utf8'))['responses']


def decode_response(survey, response):
    decoded = {}
    for k, v in response.items():
        col = survey['exportColumnMap'].get(k)
        if col is None:
            decoded[k] = v
            continue
        question_text = survey['questions'][col['question']]['questionText']
        choice = col.get('choice', None)
        if choice is not None:
            choice_val = survey['questions']
            while choice:
                if '.' in choice:
                    cur, choice = (choice).split('.', 1)
                else:
                    cur = choice
                    choice = None
                choice_val = choice_val[cur]
            question_text = '{}:{}'.format(question_text, choice_val['choiceText'])
        print(question_text, v)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_path',
                        help='Path to write surveys',
                        default='surveys')
    args = parser.parse_args()

    download_surveys(args.out_path)