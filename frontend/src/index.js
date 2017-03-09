import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

let topLevel;
if (window.location.search.slice(1, 7) === 'panopt') {
  let Panopticon = require('./Panopticon').default;
  topLevel = <Panopticon />;
} else if (window.location.search.slice(1) === 'showall') {
  let ShowAllScreens = require('./ShowAllScreens').default;
  topLevel = <ShowAllScreens />;
} else {
  let App = require('./App').default;
  topLevel = <App />;
}


ReactDOM.render(
  topLevel,
  document.getElementById('root')
);
