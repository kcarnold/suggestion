var procData = _.map(taskData, function(pair, pairIdx) {
  var rng = new Math.seedrandom(pair.context);
  var swap = rng() < .5;
  var left = swap ? pair.true_follows : pair.sugg;
  var right = (!swap) ? pair.true_follows : pair.sugg;
  return _.assign({}, pair, {pairIdx: pairIdx, selected: null, left: left, right: right});
});

var app = new Vue({
    el: "#app",
    data: {
        pairs: procData
    }
});
