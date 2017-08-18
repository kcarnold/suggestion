import {processLog} from './src/Analyzer.js';
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
  var log = res.split('\n').filter(line => line.length > 0).map(line => JSON.parse(line));
  process.stdout.write(JSON.stringify(processLog(log)));
});
