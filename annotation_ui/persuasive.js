var procData = _.map(taskData, function(text, textIdx) {
    return {textIdx: textIdx, ...text, votes: 0};
});

var app = new Vue({
    el: "#app",
    data: {
        texts: procData
    }
});
