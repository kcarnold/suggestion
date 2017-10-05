var procData = _.map(taskData, function(pair, pairIdx) {
  let swap = Math.random() < 0.5;
  let texts = _.map(pair.texts, function(text, i) {
    return _.assign({ idx: i }, text);
  });
  if (swap) {
    texts = [texts[1], texts[0]];
  }
  return {
    pairIdx: pairIdx,
    meta: pair.meta,
    swap: swap,
    texts: texts,
    selected: null,
    check_texts: pair.check_texts,
    check_selected: null
  };
});

var app = new Vue({
  el: "#app",
  data: {
    pairs: procData,
  },
});
