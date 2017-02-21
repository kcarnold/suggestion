import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
// unused: observable, action, toJS, transaction
import {observer, inject, Provider} from 'mobx-react';
import WSClient from './wsclient';
import {Keyboard} from './Keyboard';
import {ExperimentStateStore} from './ExperimentState';

//var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);
var ws = new WSClient(`ws://${window.location.host}/ws`);

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

var eventHandlers = [];

function registerHandler(fn) {
  eventHandlers.push(fn);
}

function dispatch(event) {
  console.log(event);
  log(event);
  event.timestamp = +new Date();
  eventHandlers.forEach(fn => fn(event));
}

// Every event gets logged to the server. Keep events small!
function log(event) {
  ws.send({type: 'log', event});
}


let START_PAGE = 'experiment';


class MasterStateStore {
  constructor() {
    this.__version__ = 1;
    M.extendObservable(this, {
      block: 0,
      page: START_PAGE,
      experimentState: new ExperimentStateStore(),
      get suggestionRequestParams() {
        return {
          rare_word_bonus: this.block === 0 ? 1 : 0.,
          domain: 'yelp_train'
        };
      }
    });
  }

  handleEvent = (event) => {
    if (this.experimentState) {
      this.experimentState.handleEvent(event);
    }
    switch (event.type) {
    case 'typingDone':
      this.page = 'edit';
      break;
    case 'editingDone':
      if (this.block === 0) {
        this.block = 1;
        this.experimentState = new ExperimentStateStore();
        this.page = 'experiment';
      } else {
        this.page = 'postSurvey';
      }
      break;
    default:
    }
  }
}

var state = new MasterStateStore();
registerHandler(state.handleEvent);


// Auto-runner to watch the context and request suggestions.
M.autorun(() => {
  let {experimentState} = state;
  if (!experimentState)
    return;

  let seqNum = experimentState.contextSequenceNum;

  // The only dependency is contextSequenceNum; other details don't matter.
  M.untracked(() => {
    let context = experimentState.getSuggestionContext();
    let {prefix, curWord} = context;
    ws.send({
      type: 'requestSuggestions',
      request_id: seqNum,
      sofar: prefix,
      cur_word: curWord,
      ...state.suggestionRequestParams
    });
  });
});


ws.onmessage = function(msg) {
  if (msg.type === 'suggestions') {
    dispatch({type: 'receivedSuggestions', msg});
  }
};


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

class Suggestion extends Component {
  render() {
    let {onTap, word, preview, isValid} = this.props;
    return <div
      className={"Suggestion" + (isValid ? '' : ' invalid')}
      onClick={isValid ? onTap : null}
      onTouchStart={isValid ? onTap : null}>
      {word}<span className="preview">{preview.join(' ')}</span>
    </div>;
  }
}

const SuggestionsBar = inject('state', 'dispatch')(observer(class SuggestionsBar extends Component {
  render() {
    const {state, dispatch} = this.props;
    return <div className="SuggestionsBar">
      {state.visibleSuggestions.map((sugg, i) => <Suggestion
        key={i}
        onTap={(evt) => {
          dispatch({type: 'tapSuggestion', slot: i});
          evt.preventDefault();
          evt.stopPropagation();
        }}
        word={sugg.words[0]}
        preview={sugg.words.slice(1)}
        isValid={sugg.isValid} />
      )}
    </div>;
  }
}));

const ExperimentScreen = inject('state', 'dispatch')(observer(class ExperimentScreen extends Component {
  render() {
    let {state} = this.props;
    let {experimentState} = state;
    return <Provider state={experimentState} appState={state}>
      <div className="ExperimentScreen">
      <div style={{backgroundColor: '#ccc', color: 'black'}}>
        <button onClick={evt => {
          if(confirm("Are you sure you're done?")) {
            dispatch({type: 'typingDone'});
          }
        }}>Done</button>
      </div>
      <div className="CurText">{experimentState.curText}<span className="Cursor"></span>
      </div>
      <SuggestionsBar />
      <Keyboard dispatch={this.props.dispatch} />
    </div>
    </Provider>;
  }
}));

class EditingControl extends Component {
  componentDidMount() {
    this.elt.value = this.props.initialValue;
  }
  render() {
    return <textarea ref={elt => {this.elt = elt}} />;
  }
}

const App = observer(class App extends Component {
  render() {
    let screen;
    switch(state.page) {
      case 'experiment':
        screen = <ExperimentScreen />;
        break;
      case 'edit':
        screen = <div className="EditPage">
          <div style={{backgroundColor: '#ccc', color: 'black'}}>
            Now, edit what you wrote to make it better. When you're done, tap
            <button onClick={evt => {
              if (confirm("Are you sure you're done?")) {
                dispatch({type: "editingDone"});
              }
            }}>Done</button>
          </div>
          <EditingControl initialValue={state.experimentState.curText} />
        </div>;
        break;
      case 'done':
        screen = <div>
          Thanks! Your code is {clientId}.
        </div>;
        break;
      default:
        debugger;
    }
    return (
      <Provider state={state} dispatch={dispatch}>
      <div className="App">
        {screen}
        <div className="clientId">{clientId}</div>
      </div>
      </Provider>
    );
  }
});

export default App;

// Globals
window.M = M;
window.state = state;
