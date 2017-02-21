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
  var lastText = state.experimentState.curText;
  log.forEach((entry) => {
    state.handleEvent(entry);
    if (state.experimentState.curText !== lastText) {
      annotated.push({...entry, curText: state.experimentState.curText});
      requests.push({...state.experimentState.getSuggestionContext(), ...state.suggestionRequestParams});
      lastText = state.curText;
    }
  });
  process.stdout.write(JSON.stringify({annotated, requests}));
});
