import React, { Component } from 'react';
// import logo from './logo.svg';
import './App.css';
import _ from 'lodash';

var KEYLABELS = {
    ' ': 'space'
//      '⏎': 'Done'
};

class Keyboard extends Component {
  render() {
    return <div className="Keyboard">{
      ['qwertyuiop', 'asdfghjkl', '\'?zxcvbnm⌫', '-!, .⏎'].map(function(row, i) {
          return <div key={i} className="row">{
            _.map(row, function(key, j) {
              // if (layer === 'upper') key = key.toUpperCase();
              var label = key in KEYLABELS ? KEYLABELS[key] : key;
              var className = 'key';
              if ('⏎⌫\'-!,.?'.indexOf(key) !== -1) className += ' key-reverse';
              return <div key={key} className={className} data-key={key}>{label}</div>;
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
