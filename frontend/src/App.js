import React, { Component } from 'react';
import _ from 'lodash';
import * as M from 'mobx';
import {observer, Provider} from 'mobx-react';
import WSClient from './wsclient';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './MasterView';
import Raven from 'raven-js';
import * as WSPinger from './WSPinger';

const MAX_PING_TIME = 200;

export function init(clientId, clientKind) {

  var wsURL = `ws://${window.location.host}`;
  //var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);
  var ws = new WSClient(wsURL + '/ws');

  var state = new MasterStateStore(clientId || '');


  var messageCount = {};
  window.messageCount = messageCount;

  var browserMeta = {
    userAgent: navigator.userAgent,
    screen: _.fromPairs(_.map('height availHeight width availWidth'.split(' '), x => [x, window.screen[x]])),
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
      git_rev: process.env.REACT_APP_GIT_REV,
      messageCount: messageCount,
      masterConfig: state.masterConfig,
    }]);
  }

  function addLogEntry(kind, event) {
    if (!messageCount[kind])
      messageCount[kind] = 0;
    messageCount[kind] += 1;
    updateBacklog();
  }

  if (clientId) {
    updateBacklog();
    ws.connect();
  }

  var eventHandlers = [];

  function registerHandler(fn) {
    eventHandlers.push(fn);
  }

  function _dispatch(event) {
    Raven.captureBreadcrumb({
      category: 'dispatch',
      message: event.type,
      data: event
    });
    console.log(event);
    event.jsTimestamp = +new Date();
    event.kind = clientKind;
    log(event);
    let sideEffects = [];
    eventHandlers.forEach(fn => {
      let res = fn(event);
      if (res.length) {
        sideEffects = sideEffects.concat(res);
      }
    });
    // Run side-effects after all handlers have had at it.
    sideEffects.forEach(sideEffect => {
      if (sideEffect.type === 'requestSuggestions') {
        console.log(sideEffect)
        ws.send(sideEffect);
      } else {
        if (sideEffect.type !== 'suggestion_context_changed') {
          setTimeout(() => dispatch(sideEffect), 0);
        }
      }
    });
  }

  let dispatch;
  if (process.env.NODE_ENV === 'production') {
    dispatch = (event) => {
      try {
        return _dispatch(event);
      } catch (e) {
        Raven.captureException(e, {
          tags: {dispatcher: 'dispatch'},
          extra: event
        });
        throw e;
      }
    };
  } else {
    dispatch = _dispatch;
  }

  // Every event gets logged to the server. Keep events small!
  function log(event) {
    ws.send({type: 'log', event});
    addLogEntry(clientKind, event);
  }


  registerHandler(state.handleEvent);

  var didInit = false;

  ws.onmessage = function(msg) {
    if (msg.type === 'suggestions') {
      dispatch({type: 'receivedSuggestions', msg});
    } else if (msg.type === 'backlog') {
      console.log('Backlog', msg.body.length);
      let firstTime = !didInit;
      state.replaying = true;
      msg.body.forEach(msg => {
        try {
          state.handleEvent(msg);
          addLogEntry(msg.kind, msg);
        } catch (e) {
          Raven.captureException(e, {
            tags: {dispatcher: 'backlog'},
            extra: msg
          });
          throw e;
        }
      });
      state.replaying = false;
      updateBacklog();
      if (firstTime) {
        afterFirstMessage();
        didInit = true;
      }
    } else if (msg.type === 'otherEvent') {
      console.log('otherEvent', msg.event);
      // Keep all the clients in lock-step.
      state.handleEvent(msg.event);
      addLogEntry(msg.event.kind, msg.event);
    }
  };

  // The handler for the first backlog message calls 'afterFirstMessage'.
  function afterFirstMessage() {
    if (clientKind === 'p') {
      setSizeDebounced();
    }
    if (state.pingTime === null || state.pingTime > MAX_PING_TIME) {
      setTimeout(() => WSPinger.doPing(wsURL + '/ping', 5, function(ping) {
        dispatch({type: 'pingResults', ping});
      }), 100);
    }
  }

  function setSize() {
    let width = Math.min(document.documentElement.clientWidth, window.screen.availWidth);
    let height = Math.min(document.documentElement.clientHeight, window.screen.availHeight);
    dispatch({type: 'resized', width, height});
  }

  var setSizeDebounced = _.throttle(setSize, 100, {leading: false, trailing: true});

  window.addEventListener('resize', setSizeDebounced);


  // Globals
  window.M = M;
  window.state = state;
  window.dispatch = dispatch;

  return {state, dispatch, clientId, clientKind};
}

const App = observer(class App extends Component {
  render() {
    let {state, dispatch, clientId, clientKind} = this.props.global;
    if (clientKind === 'p') {
      if (state.pingTime === null) {
        return <div>Please wait while we test your phone's communication with our server.</div>;
      } else if (state.pingTime > MAX_PING_TIME) {
        return <div>Sorry, your phone's connection to our server is too slow (your ping is {Math.round(state.pingTime)} ms). Check your WiFi connection and reload the page.</div>;
      }
    }
    return (
      <Provider state={state} dispatch={dispatch} clientId={clientId} clientKind={clientKind} spying={false}>
        <MasterView kind={clientKind} />
      </Provider>
    );
  }
});

export default App;
