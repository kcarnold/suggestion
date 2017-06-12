var MasterStateStore = require('./src/MasterStateStore').MasterStateStore;
var fs = require('fs');

console.log = console.warn;

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
        byExpPage: {},
      };
      participants.set(participant_id, experiment);
    }
    let {state, byExpPage} = experiment;
    let lastText = (state.experimentState || {}).curText;
    let isValidSugUpdate = entry.request_id === (state.experimentState || {}).contextSequenceNum;
    state.handleEvent(entry);
    if (state.experimentState) {
      let expState = state.experimentState;
      let page = state.curExperiment;
      let pageData = byExpPage[page];
      if (!pageData) {
        pageData = {
          annotated: [],
          requests: [],
          displayedSuggs: [],
        };
        byExpPage[page] = pageData;
      }
      if (expState.curText !== lastText) {
        pageData.annotated.push({...entry, curText: lastText});
        pageData.requests.push({...state.suggestionRequest});
        lastText = state.curText;
        if (pageData.displayedSuggs.length > 0) {
          let lastDisplayedSugg = pageData.displayedSuggs[pageData.displayedSuggs.length - 1];
          lastDisplayedSugg.dieEvent = event;
        }
      } else if (entry.type === 'receivedSuggestions' && isValidSugUpdate) {
        experiment.displayedSuggs.push({
          visibleSuggestions: expState.visibleSuggestions,
          displayEvent: entry,
          dieEvent: null,
        });
      }
    }
  });
  // Summarize the sub-experiments.
  participants = [...participants].map(([participant_id, {state, byExpPage}]) => ([participant_id, {
    config: state.masterConfigName,
    byExpPage,
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
