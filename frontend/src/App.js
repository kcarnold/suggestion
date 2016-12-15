import React, { Component } from 'react';
// import logo from './logo.svg';
import './App.css';
import _ from 'lodash';

var KEYLABELS = {
    ' ': 'space'
};

class Keyboard extends Component {
  lastKbdRect = null;

  handleClick = (evt) => {
    let {top, left, width, height} = this.node.getBoundingClientRect();
    let kbdRect = {top, left, width, height};
    if (!_.isEqual(kbdRect, this.lastKbdRect)) {
      this.lastKbdRect = kbdRect;
      var keyRects = {};
      this.keyRects = keyRects;
      _.forOwn(this.keyNodes, (node, key) => {
        let {top, left, width, height} = node.getBoundingClientRect();
        this.keyRects[key] = {top, left, width, height};
      });
      debugger
    }
  };

  render() {
    var keyNodes = {};
    this.keyNodes = keyNodes;
    return <div className="Keyboard" ref={node => this.node = node} onClick={this.handleClick}>{
      ['qwertyuiop', 'asdfghjkl', '\'?zxcvbnm⌫', '-!, .⏎'].map(function(row, i) {
          return <div key={i} className="row">{
            _.map(row, function(key, j) {
              // if (layer === 'upper') key = key.toUpperCase();
              var label = key in KEYLABELS ? KEYLABELS[key] : key;
              var className = 'key';
              if ('⏎⌫\'-!,.?'.indexOf(key) !== -1) className += ' key-reverse';
              return <div key={key} className={className} data-key={key} ref={node => keyNodes[key] = node}>{label}</div>;
          })}</div>
          })}
      </div>;
  }
}

class App extends Component {
  render() {
    return (
      <div className="App">
        <Keyboard />
      </div>
    );
  }
}

export default App;
