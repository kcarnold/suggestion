var procData = _.map(taskData, function(text, textIdx) {
    return {textIdx: textIdx, meta: text[0], review: _.map(text[1].slice(1), x=>x.sentence).join(' '), votes: 0};
});

var app = new Vue({
    el: "#app",
    data: {
        texts: procData
    }
});
