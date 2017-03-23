var MasterStateStore = require('./src/MasterStateStore').MasterStateStore;
var fs = require('fs');

// see https://gist.github.com/kristopherjohnson/5065599
function readStdin(callback) {
  var data = '';
  var stdin = process.stdin;
  stdin.resume();
  stdin.setEncoding('utf8');
  stdin.on('data', function (chunk) { data += chunk; });
  stdin.on('end', function() {
    callback(null, data);
  });
}

readStdin(function(err, res) {
  var participants = new Map();
  var log = res.split('\n').filter(line => line.length > 0).map(line => JSON.parse(line));
  log.forEach((entry) => {
    if (entry.kind === 'meta') return;
    let {participant_id} = entry;
    let experiment = participants.get(participant_id);
    if (!experiment) {
      experiment = {
        state: new MasterStateStore(participant_id),
        annotated: [],
        requests: [],
      };
      participants.set(participant_id, experiment);
    }
    let {state, annotated, requests} = experiment;
    let lastText = (state.experimentState || {}).curText;
    state.handleEvent(entry);
    if (state.experimentState && state.experimentState.curText !== lastText) {
      annotated.push({...entry, curText: state.experimentState.curText});
      requests.push({...state.suggestionRequest, page: state.curExperiment});
      lastText = state.curText;
    }
  });
  // Summarize the sub-experiments.
  participants = [...participants].map(([participant_id, {state, annotated, requests}]) => ([participant_id, {
    config: state.masterConfigName,
    annotated, requests,
    screenTimes: state.screenTimes,
    conditions: state.conditions,
    blocks: [0, 1].map(block => ({
      condition: state.conditions[block],
      prewriteText: (state.experiments.get(`pre-${block}`) || {}).curText,
      finalText: (state.experiments.get(`final-${block}`) || {}).curText,
      place: state.places[block],
    }))
  }]));

  process.stdout.write(JSON.stringify(participants));
});
