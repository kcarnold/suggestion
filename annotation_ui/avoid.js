var procData = _.map(taskData, function(text, textIdx) {
    return {textIdx: textIdx, ...text, votes: null};
});

var app = new Vue({
    el: "#app",
    data: {
        texts: procData
    }
});
