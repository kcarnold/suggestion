var condition = window.localStorage.hintsCondition;
if (!condition) {
    condition = window.localStorage.hintsCondition = _.sample(['examples', 'full', 'selected']);
}
var showSuggs = condition !== 'examples';
$('.expzone').addClass('cond-'+(showSuggs ? 'suggs' : 'examples'));

var writingTA = document.querySelector('[name=writing]');
$('[name=condition]').val(condition);

var logEntries = [];
function log(evt) {
    evt.timestamp = +new Date();
    logEntries.push(evt);
    $('[name=log]').val(JSON.stringify(logEntries));
}

function showHints(hints) {
    log({type: 'showHints', hints: hints});
    var box = document.querySelector('.suggs select');
    box.innerHTML = hints.map(x => "<option>"+x+"</option>").join('');
}

var prevNumFinishedSents = null;

$('.suggs select').on('change', function(evt) {
    evt.preventDefault();
    log({type: 'clickHint', hint: evt.target.value});
    writingTA.value += evt.target.value + ' ';
    evt.target.value = '';
    $(writingTA).focus();
    textUpdated();
    return false;
});

function countWords(str) {
    var matches = str.match(/\S+/g);
    return matches ? matches.length : 0;
}

function textUpdated() {
    var curText = writingTA.value;
    $('#wordcount').text(countWords(curText));
    var numFinishedSents = (curText.replace(/\.{3,}/, ' _ELLIPS_ ').match(/[\.\?\!]+/g) || []).length;
    if (numFinishedSents !== prevNumFinishedSents) {
        console.log(numFinishedSents);
        if (condition === 'selected') {
            showHints(getAppropriateHints(numFinishedSents, 10));
        }
    }
    prevNumFinishedSents = numFinishedSents;
}

$(writingTA).on('input', function(evt) {
    textUpdated();
});

function getAppropriateHints(numSents, max) {
    var hints = [];
    var sentPosteriorIdx = numSents > 2 ? 2 : numSents;
    clusters.forEach(function(group) {
        var options = group.filter(function(pair) {
            var prob = pair[0][sentPosteriorIdx];
            return prob > .5;
        });
        if (options.length > 0)
            hints.push(_.sample(options)[1]);
    });
    return _.shuffle(hints).slice(0, max);
}

if (condition === 'examples') {
    var ex = $('.examples');
    ex.empty();
    examples.forEach(function(example) {
        var elt = $("<div class='example'>");
        elt.text(example);
        ex.append(elt);
    });
} else if (condition === 'full') {
    showHints(_.shuffle(clusters.map(function(group) {
        return _.sample(group)[1];
    })));
}

textUpdated();
