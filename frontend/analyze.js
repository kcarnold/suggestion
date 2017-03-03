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
  var log = res.split('\n').filter(line => line.length > 0).map(line => JSON.parse(line));
  var state = new MasterStateStore();
  var annotated = [];
  var requests = [];
  var lastText = (state.experimentState || {}).curText;
  var curTrial = {};
  var trials = [];
  var prevScreen = null;
  let pageDurs = [];
  let pageStartTime = null;
  log.forEach((entry) => {
    state.handleEvent(entry);
    if (state.experimentState && state.experimentState.curText !== lastText) {
      annotated.push({...entry, curText: state.experimentState.curText});
      requests.push(state.suggestionRequest);
      lastText = state.curText;
    }
    let curScreen = state.curScreen;
    if (curScreen !== prevScreen) {
      // Page transition.
      // - record page durations
      if (pageStartTime !== null) {
        pageDurs.push([prevScreen.screen || prevScreen.controllerScreen, (entry.jsTimestamp - pageStartTime) / 1000]);
      }
      pageStartTime = entry.jsTimestamp;

      // - record writing outcomes.
      if (curScreen.controllerScreen === 'PostTaskSurvey') {
        // Record the trial data.
        curTrial.block = state.block;
        curTrial.conditionName = state.conditionName;
        let condition = state.condition;
        curTrial.showPhrase = condition.showPhrase;
        curTrial.prewriteText = state.experimentState.curText;
        curTrial.editedText = state.curEditText;
        trials.push(curTrial);
        curTrial = {};
      }
      prevScreen = curScreen;
    }
  });
  process.stdout.write(JSON.stringify({annotated, requests, trials, pageDurs}));
});
