import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

let isPanopticon = window.location.search.slice(1, 7) === 'panopt';
let topLevel;
if (isPanopticon) {
  let Panopticon = require('./Panopticon').default;
  topLevel = <Panopticon />;
} else {
  let App = require('./App').default;
  topLevel = <App />;
}


ReactDOM.render(
  topLevel,
  document.getElementById('root')
);
