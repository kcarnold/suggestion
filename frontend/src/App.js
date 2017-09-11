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
const defaultConfig = 'study1';

export function init() {
  let externalAction  = window.location.hash.slice(1);
  window.location.hash = '';

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
    let code = params === 'new' ? '' : prompt("If you have a code alreday, enter it here, otherwise just press OK:");
    let hash = '';
    if (!code) {
      // Generate a code.
      clientId = _.range(6).map(function(i) { return _.sample('23456789cfghjmpqrvwx'); }).join('');
      code = clientId + '-p';
      let config = defaultConfig;
      if (externalAction.slice(0, 2) === 'c=') {
        config = externalAction.slice(2);
      }
      hash = `#c=${config}`;
    }
    code = code.toLowerCase();
    window.location.replace(`${window.location.protocol}//${window.location.host}/?${code}${hash}`);
    // That should cause a reload, once the rest of this script finishes.
    return [null, null];
  })();


  var wsURL = `ws://${window.location.host}`;
  //var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);
  var ws = new WSClient(wsURL + '/ws');

  var state = new MasterStateStore(clientId || '');


  var logs = {};
  window.logs = logs;

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
      messageCount: _.mapValues(logs, v => v.length),
      masterConfig: state.masterConfig,
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
        Raven.captureException(e);
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
      console.log('Backlog', msg);
      let firstTime = !didInit;
      state.replaying = true;
      msg.body.forEach(msg => {
        try {
          state.handleEvent(msg);
          addLogEntry(msg.kind, msg);
        } catch (e) {
          Raven.captureException(e)
          throw e;
        }
      });
      // This needs to happen here so that we don't temporarily display the redirect page.
      if (externalAction) {
        dispatch({type: 'externalAction', externalAction});
        externalAction = '';
      }
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
      setSize();
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
    if (height < 450) {
      if (width > height) {
        // alert('Please rotate your phone to be in the portrait orientation.');
      } else {
        alert("Your screen is small; things might not work well.");
      }
    }
    dispatch({type: 'resized', width, height});
  }

  window.addEventListener('resize', function() {
      setTimeout(setSize, 10);
  });


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
