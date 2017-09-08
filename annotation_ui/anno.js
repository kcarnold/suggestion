var procData = _.map(taskData, function(text, textIdx) {
    return {textIdx: textIdx, meta: text[0], data: _.map(text[1].slice(1), function(sent, sentIdx) {
        return _.assign({sentIdx: sentIdx, pos: null, neg: null, nonsense: false}, sent);
    })};
});

var app = new Vue({
    el: "#app",
    data: {
        texts: procData
    }
});
