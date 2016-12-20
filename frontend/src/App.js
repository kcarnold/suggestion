import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
// unused: observable, action, toJS, transaction
import {observer, inject, Provider} from 'mobx-react';
import WSClient from './wsclient';

var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);

// Generate a hopefully-unique id
var clientId = null;
if (window.location.search.match(/^\?\d{6}$/)) {
    clientId = window.location.search.slice(1);
} else if (window.localStorage.getItem('clientId')) {
    clientId = window.localStorage.getItem('clientId');
} else {
    clientId = _.range(6).map(function(i) { return _.sample('0123456789abcdef'); }).join('');
}
window.localStorage['clientId'] = clientId;
ws.sendHello({type: 'init', participantId: clientId});


/**** Event dispatching

This is how we split the difference between Flux everything-is-a-big-global-action and mobx just-mutate-stuff:
All input comes in as events represented as plain JSON objects. The level of interpretation should be pragmatic:
low-level enough to be able to get fine-grained detail about what happened, but high-level enough to be able
to read off interesting things without much work. e.g., for a key tap, include the tap x/y position, but also
what key we thought it was.

All server communication comes in this way too.
 */

// Dispatch by type. This was probably premature...
var handlersByType = {};

function registerHandler(eventType, fn) {
  handlersByType[eventType] = handlersByType[eventType] || [];
  handlersByType[eventType].push(fn);
}

function dispatch(event) {
  console.log(event);
  log(event);
  event.timestamp = +new Date();
  let handlers = handlersByType[event.type];
  if (!handlers) {
    console.warn('Dispatched event with no handlers', event);
  } else {
    handlers.forEach(fn => fn(event));
  }
}

// Every event gets logged to the server. Keep events small!
function log(event) {
  ws.send({type: 'log', event});
}


/**** Main state store!

This represents all the state, UI state included. The entire UI should be a function of this, with perhaps a few pragmatic exceptions (none yet - 2016-12-17).

A Suggestion is an important object. It contains:
{
  // The context in which it was made
  contextSequenceNum: int
  prefix: string : characters before this suggestion
  tapLocations: array of {x: int, y: int}: cur-word taps
  words: array of string: the words of the suggestion
  probs: array of float: the natural-log generation probabilities of each word
  isValid: for a visible suggestion, whether it's valid in the current suggestion context.
}

We keep a list of the current suggestions. If a suggestion's sequence number doesn't match, we can still show it (so as to not be jarring), but dim it, and don't allow acceptance.

After tapping a suggestion, it becomes *active*, which means that if we keep tapping on that same spot, subsequent words will get inserted. To do this, we keep a notion of an `activeSuggestion`, which is separate from the suggestions returned from the server:
{
  suggestion: the Suggestion object
  slot: the slot number where it lives
  wordIdx: the index of the next word to insert.
}

We're not caching because (1) caching is a Hard Problem and (2) key taps will be at different places anyway, so caching won't really help.
It may be nice to prefetch what happens when a suggestion gets pressed, because there are only 3 states there and we know them exactly.
But TODO.

How this gets updated:
An autorun thread watches contextSequenceNum and requests suggestions when the context changes.

visibleSuggestions is a pure computation based on the last suggestions received from the server, the current context sequence number, and the active suggestion. It puts them all together, keeping track of what's valid.

*/
class StateStore {
  constructor() {
    this.__version__ = 1;
    M.extendObservable(this, {
      curText: '',
      tapLocations: [],
      contextSequenceNum: 0,
      lastSuggestionsFromServer: null,
      activeSuggestion: null,
      get activeSuggestionWords() {
        return this.activeSuggestion.suggestion.words.slice(this.activeSuggestion.wordIdx);
      },
      get visibleSuggestions() {
        // The visible suggestions are:
        // - The active ("promised") suggestion, in its corresponding slot.
        // - Suggestions from the server, filled in as long as they're valid.

        // Copy the suggestions so we can tweak them.
        let suggestions = M.toJS(this.lastSuggestionsFromServer);

        if (this.activeSuggestion) {
          suggestions[this.activeSuggestion.slot] = {
            orig: this.activeSuggestion.suggestion,
            contextSequenceNum: this.contextSequenceNum,
            words: this.activeSuggestionWords,
          }
        }
        return suggestions;
      },
      tapKey: M.action(event => {
        this.curText += event.key;
        this.tapLocations.push({x: event.x, y: event.y});
        this.contextSequenceNum++;
      }),
      tapBackspace: M.action(() => {
        this.curText = this.curText.slice(0, -1);
        this.tapLocations.pop();
        this.contextSequenceNum++;
      }),
      insertSuggestion: M.action(slot => {
        let wordToInsert = null;
        let tappedSuggestion = this.visibleSuggestions[slot];
        if (this.activeSuggestion !== null && this.activeSuggestion.slot === slot) {
          // Continuing a previous suggestion.
          wordToInsert = this.activeSuggestionWords[0];
          if (this.activeSuggestionWords.length > 1) {
            // Words remain, advance the active suggestion.
            this.activeSuggestion.wordIdx++;
          } else {
            // No more words in the active suggestion, go back to normal.
            this.activeSuggestion = null;
          }
        } else if (tappedSuggestion.contextSequenceNum === this.contextSequenceNum) {
          wordToInsert = tappedSuggestion.words[0];
          if (tappedSuggestion.words.length > 1) {
            this.activeSuggestion = {
              suggestion: tappedSuggestion,
              slot: slot,
              wordIdx: 1
            };
          }
        }
        let {prefix} = this.getSuggestionContext();
        this.curText = prefix + wordToInsert;
        this.tapLocations = this.tapLocations.slice(0, prefix.length).concat(wordToInsert.map(() => null));
      }),
    });
  }

  getSuggestionContext() {
    let sofar = this.curText, cursorPos = sofar.length;
    let lastSpaceIdx = sofar.search(/\s\S*$/);
    let curWord = [];
    for (let i=lastSpaceIdx + 1; i<cursorPos; i++) {
      let chr = {letter: sofar[i]};
      if (this.tapLocations[i] !== null) {
        chr.tap = this.tapLocations[i];
      }
      curWord.push(chr);
    }
    return {
      prefix: sofar.slice(0, lastSpaceIdx + 1),
      curWord
    };
  }
}

var state = new StateStore();

// Auto-runner to watch the context and request suggestions.
M.autorun(() => {
  let context = M.untracked(() => state.getSuggestionContext());
  let {prefix, curWord} = context;
  ws.send({
    type: 'requestSuggestions',
    request_id: state.contextSequenceNum,
    sofar: prefix,
    cur_word: curWord,
    temperature: .5,
    domain: 'yelp_train',
  });
});

ws.onmessage = function(msg) {
  if (msg.type === 'suggestions') {
    dispatch({type: 'receivedSuggestions', msg});
  }
};


registerHandler('tapKey', event => {
  state.tapKey(event);
});

registerHandler('tapBackspace', event => {
  state.tapBackspace();
});

registerHandler('receivedSuggestions', ({msg}) => {
  // Only update suggestions if the data is valid.
  if (msg.request_id !== state.contextSequenceNum) {
    console.log("Discarding outdated suggestions", msg.request_id, state.contextSequenceNum);
    return;
  }

  let suggestions = msg.next_word.map(sugg => ({
    orig: sugg,
    contextSequenceNum: msg.request_id,
    words: sugg.one_word.words.concat(sugg.continuation.length ? sugg.continuation[0].words : []),
    probs: sugg.probs,
  }));

  // Reorder the suggestions to match the active suggestion, if applicable.
  if (state.activeSuggestion !== null) {
    let activeSuggestionNextWord = state.activeSuggestionWords[0];
    let nextWordIdx = _.map(suggestions, sugg => sugg.words[0]).indexOf(activeSuggestionNextWord);
    if (nextWordIdx !== -1) {
      // Move the corresponding server suggestion to the correct spot.
      suggestions.splice(state.activeSuggestion.slot, 0, suggestions.splice(nextWordIdx, 1)[0]);
    } else {
      // not one of the next words. Leave a blank for it.
      suggestions.splice(state.activeSuggestion.slot, 0, null);
    }
  }

  state.lastSuggestionsFromServer = suggestions;
});

var KEYLABELS = {
    ' ': 'space',
    '\n': '⏎',
};

function getClosestKey(keyRects, touchX, touchY) {
    var closestKey = null, closestDist = Infinity;
    keyRects.forEach(function(krect) {
        var rect = krect.rect, hwidth = rect.width / 2, hheight = rect.height / 2, x = rect.left + hwidth, y = rect.top + hheight;
        var dx = Math.max(0, Math.abs(touchX - x) - hwidth), dy = Math.max(0, Math.abs(touchY - y) - hheight);
        var dist = dx * dx + dy * dy;
        if (dist < closestDist) {
            closestDist = dist;
            closestKey = krect.key;
        }
    });
    return closestKey;
}

class Keyboard extends Component {
  lastKbdRect = null;

  handleClick = (evt) => {
    let {top, left, width, height} = this.node.getBoundingClientRect();
    let kbdRect = {top, left, width, height};
    if (!_.isEqual(kbdRect, this.lastKbdRect)) {
      this.lastKbdRect = kbdRect;
      var keyRects = [];
      this.keyRects = keyRects;
      _.forOwn(this.keyNodes, (node, key) => {
        let {top, left, width, height} = node.getBoundingClientRect();
        this.keyRects.push({rect: {top, left, width, height}, key});
      });
      evt.preventDefault();
      evt.stopPropagation();
    }

    let key = getClosestKey(this.keyRects, evt.clientX, evt.clientY);
    if (key === '⏎')
      key = '\n';
    if (key === '⌫') {
      dispatch({type: 'tapBackspace'});
    } else {
      dispatch({type: 'tapKey', key, x: evt.clientX, y: evt.clientY});
    }
  };

  render() {
    var keyNodes = {};
    this.keyNodes = keyNodes;
    return <div className="Keyboard" ref={node => this.node = node} onClick={this.handleClick}>{
      ['qwertyuiop', 'asdfghjkl', '\'?zxcvbnm⌫', '-!, .\n'].map(function(row, i) {
          return <div key={i} className="row">{
            _.map(row, function(key, j) {
              // if (layer === 'upper') key = key.toUpperCase();
              var label = key in KEYLABELS ? KEYLABELS[key] : key;
              var className = 'key';
              if ('\n⌫\'-!,.?'.indexOf(key) !== -1) className += ' key-reverse';
              return <div key={key} className={className} data-key={key} ref={node => keyNodes[key] = node}>{label}</div>;
          })}</div>
          })}
      </div>;
  }
}

function setSize() {
  let width = Math.min(document.documentElement.clientWidth, screen.availWidth);
  let height = Math.min(document.documentElement.clientHeight, screen.availHeight);
  if (height < 450) {
    if (width > height)
      alert('Please rotate your phone to be in the portrait orientation.');
    else
      alert("Your screen is small; things might not work well.");
  }
  dispatch({type: 'resized', width, height});
}

window.addEventListener('resize', function() {
    setTimeout(setSize, 10);
});

setSize();

const SuggestionsBar = inject('state', 'dispatch')(observer(class SuggestionsBar extends Component {
  render() {
    return <div className="SuggestionsBar">
      {[1,2,3].map((x, i) => <div key={i} className="Suggestion">sugg</div>)
      }
    </div>

  }
}));

const ExperimentScreen = inject('state', 'dispatch')(observer(class ExperimentScreen extends Component {
  render() {
    let {state} = this.props;
    return  <div className="ExperimentScreen">
      <div className="CurText">{state.curText}</div>
      <SuggestionsBar />
      <Keyboard />
    </div>;
  }
}));

class App extends Component {
  render() {
    return (
      <Provider state={state} dispatch={dispatch}>
      <div className="App">
        <ExperimentScreen />
      </div>
      </Provider>
    );
  }
}

export default App;

// Globals
window.M = M;
window.state = state;
