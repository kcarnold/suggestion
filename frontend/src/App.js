import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
import {observer, Provider} from 'mobx-react';
import WSClient from './wsclient';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './Views';


// Get client id and kind from params or asking the user.
var [clientId, clientKind] = (function() {
  let params = window.location.search.slice(1);
  let match = params.match(/^(\w+)-(\w+)$/);
  let clientId, kind;
  if (match) {
    clientId = match[1];
    kind = match[2];
    return [clientId, kind];
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

let externalAction = window.location.hash.slice(1);
window.location.hash = '';

//var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);
var ws = new WSClient(`ws://${window.location.host}/ws`);

var logs = {};
window.logs = logs;

var browserMeta = {
  userAgent: navigator.userAgent,
  screen: _.fromPairs(_.map('height availHeight width availWidth'.split(' '), x => [x, screen[x]])),
  window: {
    devicePixelRatio: window.devicePixelRatio,
  },
  documentElement: {
    clientHeight: document.documentElement.clientHeight,
    clientWidth: document.documentElement.clientWidth,
  },
};


function updateBacklog() {
  ws.setHello([{
    type: 'init',
    participantId: clientId,
    kind: clientKind,
    browserMeta,
    messageCount: _.mapValues(logs, v => v.length),
  }]);
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


var state = new MasterStateStore(clientId);
registerHandler(state.handleEvent);


function startRequestingSuggestions() {
  // Auto-runner to watch the context and request suggestions.
  M.autorun(() => {
    let {suggestionRequest} = state;
    if (!suggestionRequest)
      return;

    console.log('requesting', suggestionRequest);
    ws.send(suggestionRequest);
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
    // This needs to happen here so that we don't temporarily display the redirect page.
    if (externalAction) {
      dispatch({type: 'externalAction', externalAction});
    }
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


const App = observer(class App extends Component {
  render() {
    return (
      <Provider state={state} dispatch={dispatch} clientId={clientId} clientKind={clientKind} spying={false}>
        <MasterView kind={clientKind} />
      </Provider>
    );
  }
});

export default App;

// Globals
window.M = M;
window.state = state;
window.dispatch = dispatch;
