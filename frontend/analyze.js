var MasterStateStore = require('./src/MasterStateStore').MasterStateStore;
var fs = require('fs');

console.log = console.warn;
global.alert = () => {};

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
        pageSeq: [],
      };
      participants.set(participant_id, experiment);
    }
    let {state, byExpPage, pageSeq} = experiment;
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
          displayedSuggs: [],
          condition: state.conditionName,
        };
        byExpPage[page] = pageData;
        pageSeq.push(page);
      }
      if (expState.curText !== lastText) {
        pageData.annotated.push({...entry, curText: lastText});
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
  participants = [...participants].map(([participant_id, {state, byExpPage, pageSeq}]) => ([participant_id, {
    config: state.masterConfigName,
    byExpPage,
    pageSeq,
    screenTimes: state.screenTimes,
    conditions: state.conditions,
    blocks: state.conditions.map((condition, block) => {
      let expState = state.experiments.get(`final-${block}`) || {};
      return {
        condition: condition,
        prewriteText: (state.experiments.get(`pre-${block}`) || {}).curText,
        finalText: expState.curText,
        place: state.places[block],
        attentionCheckStats: expState.attentionCheckStats,
      };
    }),
  }]));

  process.stdout.write(JSON.stringify(participants));
});
