import React, { Component } from 'react';
// import logo from './logo.svg';
import './App.css';
import _ from 'lodash';
import {extendObservable, observable, action, autorun, toJS, transaction} from 'mobx';
import {observer, inject, Provider} from 'mobx-react';
import WSClient from './wsclient';

var ws = new WSClient(`ws://${process.env.REACT_APP_BACKEND_HOST}:${process.env.REACT_APP_BACKEND_PORT}/ws`);

var handlersByType = {};

function registerHandler(eventType, fn) {
  handlersByType[eventType] = handlersByType[eventType] || [];
  handlersByType[eventType].push(fn);
}

function dispatch(event) {
  console.log(event);
  event.timestamp = +new Date();
  let handlers = handlersByType[event.type];
  if (!handlers) {
    console.warn('Dispatched event with no handlers', event);
  } else {
    handlers.forEach(fn => fn(event));
  }
}

class StateStore {
  constructor() {
    this.__version__ = 1;
    extendObservable(this, {
      curText: 'Test text',
    });
  };
}

var state = new StateStore();
window.state = state;

registerHandler('tapKey', event => {
  let key = event.key;
  switch(key) {
    case '⌫':
      state.curText = state.curText.slice(0, -1);
      break;
    case '⏎':
      key = '\n';
  }
  state.curText = state.curText + event.key;
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
    dispatch({type: 'tapKey', key});
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
