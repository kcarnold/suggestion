import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

import Raven from 'raven-js';
if (process.env.NODE_ENV === 'production') {
  Raven
      .config('https://c0c96b3696f14e4eb2fe4f35f4da3176@sentry.io/186354')
      .config({
        release: process.env.REACT_APP_GIT_REV,
      })
      .install();
}

let topLevel;
let query = window.location.search.slice(1);

if (window.location.search.slice(1, 7) === 'panopt') {
  let Panopticon = require('./Panopticon').default;
  topLevel = <Panopticon />;
} else if (query === 'showall') {
  let ShowAllScreens = require('./ShowAllScreens').default;
  topLevel = <ShowAllScreens />;
} else if (query === 'bench') {
  let Bench = require('./Bench').default;
  topLevel = <Bench />;
} else if (query === 'demos') {
  let DemoList = require('./DemoList').default;
  topLevel = <DemoList />;
} else {
  let App = require('./App').default;
  topLevel = <App />;
}


ReactDOM.render(
  topLevel,
  document.getElementById('root')
);
