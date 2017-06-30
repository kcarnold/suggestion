import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

import Raven from 'raven-js';
Raven
    .config('https://c0c96b3696f14e4eb2fe4f35f4da3176@sentry.io/186354')
    .config({
      release: process.env.REACT_APP_GIT_REV,
    })
    .install();

let topLevel;
if (window.location.search.slice(1, 7) === 'panopt') {
  let Panopticon = require('./Panopticon').default;
  topLevel = <Panopticon />;
} else if (window.location.search.slice(1) === 'showall') {
  let ShowAllScreens = require('./ShowAllScreens').default;
  topLevel = <ShowAllScreens />;
} else if (window.location.search.slice(1) === 'bench') {
  let Bench = require('./Bench').default;
  topLevel = <Bench />;
} else {
  let App = require('./App').default;
  topLevel = <App />;
}


ReactDOM.render(
  topLevel,
  document.getElementById('root')
);
