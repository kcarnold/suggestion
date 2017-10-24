var questions = [
  ["interesting", "I found the review interesting."],
  ["imagine", "I can imagine what it would be like to go here.",],
  ["similarExperience", "My experience here would be like the writer's experience."],
  ["enjoy", "I or someone I know would enjoy an experience here.",],
  ["intendToGo", "I'd like to go here."]
];

var names = ['A', 'B'];

var procData = _.map(taskData, function(pair, pairIdx) {
  let swap = Math.random() < 0.5;
  let texts = _.map(pair.texts, function(text, i) {
    return _.assign({ idx: i, name: names[i] }, text);
  });
  if (swap) {
    texts = [texts[1], texts[0]];
  }
  return {
    pairIdx: pairIdx,
    meta: pair.meta,
    swap: swap,
    texts: texts,
    queries: _.map(questions, function(q) {
      return {
        id: q[0],
        text: q[1],
        selected: null
      };
    }),
    check_texts: pair.check_texts,
    check_selected: null
  };
});

var app = new Vue({
  el: "#app",
  data: {
    pairs: procData,
    names: names,
  },
});
