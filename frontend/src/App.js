import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
import {observer, inject, Provider} from 'mobx-react';
import StarRatingComponent from 'react-star-rating-component';
import WSClient from './wsclient';
import {Keyboard} from './Keyboard';
import {MasterStateStore} from './MasterStateStore';

// Get client id and kind from params or asking the user.
var [clientId, clientKind, externalAction] = (function() {
  let [params, externalAction] = window.location.search.slice(1).split('&');
  let match = params.match(/^(\w+)-(\w+)$/);
  let clientId, kind;
  if (match) {
    clientId = match[1];
    kind = match[2];
    return [clientId, kind, externalAction];
  }
  let code = prompt("If you have a code alreday, enter it here, otherwise just press OK:");
  if (!code) {
    // Generate a code.
    clientId = _.range(6).map(function(i) { return _.sample('0123456789abcdef'); }).join('');
    code = clientId + '-c';
  }
  window.location.search = '?' + code;
  // That should cause a reload, once the rest of this script finishes.
  return [null, null];
})();

//var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);
var ws = new WSClient(`ws://${window.location.host}/ws`);

var logs = {};
window.logs = logs;

function updateBacklog() {
  ws.setHello([{type: 'init', participantId: clientId, kind: clientKind, messageCount: _.mapValues(logs, v => v.length)}]);
}

function addLogEntry(kind, event) {
  if (!logs[kind])
    logs[kind] = [];
  logs[kind].push(event);
  updateBacklog();
}

if (clientId) {
  updateBacklog();
  ws.connect();
}

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
  event.jsTimestamp = +new Date();
  event.kind = clientKind;
  log(event);
  eventHandlers.forEach(fn => fn(event));
}

// Every event gets logged to the server. Keep events small!
function log(event) {
  ws.send({type: 'log', event});
  addLogEntry(clientKind, event);
}


var state = new MasterStateStore(clientId, clientKind);
registerHandler(state.handleEvent);


function startRequestingSuggestions() {
  // Auto-runner to watch the context and request suggestions.
  M.autorun(() => {
    let {experimentState} = state;
    if (!experimentState)
      return;

    let seqNum = experimentState.contextSequenceNum;

    // Abort if we already have the suggestions for this context.
    if (experimentState.lastSuggestionsFromServer.length > 0 &&
        experimentState.lastSuggestionsFromServer[0].contextSequenceNum === seqNum)
      return;

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
}

var didInit = false;

ws.onmessage = function(msg) {
  if (msg.type === 'suggestions') {
    dispatch({type: 'receivedSuggestions', msg});
  } else if (msg.type === 'backlog') {
    console.log('Backlog', msg);
    let firstTime = !didInit;
    state.replaying = true;
    msg.body.forEach(msg => {
      state.handleEvent(msg);
      addLogEntry(msg.kind, msg);
    });
    state.replaying = false;
    updateBacklog();
    if (firstTime) {
      init();
      didInit = true;
    }
  } else if (msg.type === 'otherEvent') {
    console.log('otherEvent', msg.event);
    // Keep all the clients in lock-step.
    state.handleEvent(msg.event);
    addLogEntry(msg.event.kind, msg.event);
  }
};

// The handler for the first backlog call 'init'.
function init() {
    if (clientKind === 'p') {
      startRequestingSuggestions();
      setSize();
    }
    if (externalAction) {
      dispatch({type: 'externalAction', externalAction});
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

class Suggestion extends Component {
  render() {
    let {onTap, word, preview, isValid} = this.props;
    return <div
      className={"Suggestion" + (isValid ? '' : ' invalid')}
      onTouchStart={isValid ? onTap : null}>
      {word}<span className="preview">{preview.join(' ')}</span>
    </div>;
  }
}

const SuggestionsBar = inject('expState', 'dispatch')(observer(class SuggestionsBar extends Component {
  render() {
    const {expState, dispatch} = this.props;
    let {showPhrase} = expState.condition;
    return <div className="SuggestionsBar">
      {expState.visibleSuggestions.map((sugg, i) => <Suggestion
        key={i}
        onTap={(evt) => {
          dispatch({type: 'tapSuggestion', slot: i});
          evt.preventDefault();
          evt.stopPropagation();
        }}
        word={sugg.words[0]}
        preview={showPhrase ? sugg.words.slice(1) : []}
        isValid={sugg.isValid} />
      )}
    </div>;
  }
}));

function advance(state, dispatch) {
  dispatch({type: 'next'})
}

const NextBtn = inject('dispatch', 'state')((props) => <button onClick={() => {
  if (!props.confirm || confirm("Are you sure?")) {
    advance(props.state, props.dispatch);
  }
  }} disabled={props.disabled}>{props.children || "Next"}</button>);

function approxTime(remain) {
  if (remain > 60)
    return '~' + Math.round(remain / 60) + ' min';
  if (remain > 10)
    return '~' + (Math.round(remain / 10) * 10) + ' sec';
  return Math.ceil(remain) + ' sec';
}

const Timer = inject('dispatch', 'state')(observer(class Timer extends Component {
  state = {remain: Infinity};
  tick = () => {
    let {dispatch, state} = this.props;
    if (!state.timerStartedAt) return;
    let elapsed = (+new Date() - state.timerStartedAt) / 1000;
    let remain = state.timerDur - elapsed;
    this.setState({remain});
    if (remain > 0) {
      this.timeout = setTimeout(this.tick, 100);
    } else {
      this.timeout = null;
      advance(state, dispatch);
    }
  };

  componentDidMount() {
    this.tick();
  }

  componentWillUnmount() {
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
  }

  render() {
    let {remain} = this.state;
    return <div className="timer">{approxTime(remain)}</div>;
  }
}));

const ControlledInput = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <input
  onChange={evt => {dispatch({type: 'controlledInputChanged', name, value: evt.target.value});}} value={state.controlledInputs.get(name) || ''} />));

const ControlledStarRating = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <StarRatingComponent
  name={name} starCount={5} value={state.controlledInputs.get(name) || 0}
  onStarClick={value => {dispatch({type: 'controlledInputChanged', name, value});}}
  renderStarIcon={(idx, value) => <i style={{fontStyle: 'normal'}}>{idx<=value ? '\u2605' : '\u2606'}</i>} />));

const screenViews = {
  Welcome: () => <div>
    <h1>Welcome</h1>
    <p>By continuing, you agree that you have been provided with the consent form
    for this study and agree to its terms.</p>
    <NextBtn /></div>,

  ProbablyWrongCode: () => <div>
    <p>Waiting for consent on computer. If you're seeing this on your phone, you probably mistyped your code.</p>
  </div>,

  SelectRestaurants: inject('state')(observer(({state}) => <div>
    <p>Think of 2 restaurants or cafes you've been to recently.</p>
    <div>1. <ControlledInput name="restaurant1"/><br />When were you last there? <ControlledInput name="visit1"/>
      <br />How would you rate that visit? <ControlledStarRating name="star1" />
    </div>
    <div>2. <ControlledInput name="restaurant2"/><br /> When were you last there? <ControlledInput name="visit2"/>
      <br />How would you rate that visit? <ControlledStarRating name="star2" />
    </div>
    <p>(The Next button will be enabled once all fields are filled out.)</p>
    <NextBtn disabled={!_.every('restaurant1 visit1 star1 restaurant2 visit2 star2'.split(' '), x => state.controlledInputs.get(x))} />
  </div>)),

  Instructions: inject('state')(observer(({state}) => <div>
    <p>Great, the phone is paired. Let's get ready for the experiment.</p>
    <h1>Instructions</h1>
    <p>Think about your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>.</p>
    <p>Let's write a review of this experience (like you might see on a site like Yelp or Google Maps). We'll do this in <b>two steps</b>:</p>
    <ol>
      <li>Type out a very rough draft. Here we won't be concerned about grammar, coherence, accuracy, etc.</li>
      <li>Edit what you wrote into a good review.</li>
    </ol>
    <p>Tap Next when you're ready to start typing the rough draft. You will have {state.nextScreen.timer / 60} minutes (note the timer on top).</p>
    <NextBtn /></div>)),

  ExperimentScreen: inject('state', 'dispatch')(observer(({state, dispatch}) => {
      let {experimentState} = state;
      return <Provider expState={experimentState}>
        <div className="ExperimentScreen">
        <div style={{backgroundColor: '#ccc', color: 'black'}}>
          Rough draft review for your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b> ({state.curPlace.stars} stars)
          <Timer />
        </div>
        <div className="CurText">{experimentState.curText}<span className="Cursor"></span>
        </div>
        <SuggestionsBar />
        <Keyboard dispatch={dispatch} />
      </div>
      </Provider>;
    })),

  BreakBeforeEdit: inject('state')(observer(({state}) => <div>
    <p>Time is up for the rough draft. Next, you'll get a chance to edit what you wrote to make it more coherent and accurate. You'll use your computer.</p>
    <p>You'll have {state.nextScreen.timer / 60} minutes to edit the review.</p>
    <NextBtn />
    </div>)),

  BreakBeforeEditPhone: () => <div>Time's up for the rough draft. Follow the instructions on your computer to continue.</div>,

  EditScreen: inject('state', 'dispatch')(observer(({state, dispatch}) => <div className="EditPage">
    <div style={{backgroundColor: '#ccc', color: 'black'}}>
      Now, edit what you wrote to make it better. <Timer />
    </div>
    <textarea value={state.curEditText} onChange={evt => {dispatch({type: 'controlledInputChanged', name: state.curEditTextName, value: evt.target.value});}} />;
  </div>)),

  PostTaskSurvey: () => <div>The post-task survey would go here. For now, just click <NextBtn /></div>,
  PostExpSurvey: () => <div>The post-experiment survey would go here. For now, just click <NextBtn /></div>,
  Done: () => <div>Thanks! Your code is {clientId}.</div>,
  LookAtPhone: () => <div><p>Complete this step on your phone.</p> If you need it, your phone code is <tt>{clientId}-p</tt>.</div>,
  LookAtComputer: () => <div><p>Complete this step on your computer.</p> If you need it, your computer code is <tt>{clientId}-c</tt>.</div>,
  SetupPairingComputer: () => <div>
    <p>For this experiment, you'll need a smartphone.</p>
    <p>On your phone's web browser, go to <tt>megacomplete.net</tt> and enter <tt>{clientId}-p</tt>. There will be a button on that page to continue.</p>
    <p>If you have a barcode reader on your phone, you can use scan this:<br/><img src={"https://zxing.org/w/chart?cht=qr&chs=350x350&chld=L&choe=UTF-8&chl=" + encodeURIComponent("http://megacomplete.net/?" + clientId + "-p")} role="presentation"/></p>
  </div>,
  SetupPairingPhone: () => <div>Successfully paired! <NextBtn /></div>,
};


const App = observer(class App extends Component {
  render() {
    if (state.replaying) return <div>Loading...</div>;
    let screenName;
    let screenDesc = state.screens[state.screenNum];
    if (clientKind === 'c') {
      screenName = screenDesc.controllerScreen || 'LookAtPhone';
    } else {
      screenName = screenDesc.screen || 'LookAtComputer';
    }

    return (
      <Provider state={state} dispatch={dispatch}>
        <div className="App">
          {React.createElement(screenViews[screenName])}
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
window.dispatch = dispatch;