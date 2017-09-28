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
let initialPart = query.split('-', 1)[0] || query;

if (initialPart === 'panopt') {
  let Panopticon = require('./Panopticon').default;
  topLevel = <Panopticon />;

} else if (initialPart === 'showall') {
  let mod = require('./ShowAllScreens');
  mod.init(query.slice(initialPart.length + 1));
  let ShowAllScreens = mod.default;
  topLevel = <ShowAllScreens />;

} else if (initialPart === 'bench') {
  let Bench = require('./Bench').default;
  topLevel = <Bench />;

} else if (initialPart === 'demos') {
  let DemoList = require('./DemoList').default;
  topLevel = <DemoList />;

} else if (query.slice(0, 3) === 'new') {
  let params = query.split('&').slice(1).map(x=>x.split('='));
  topLevel = <div>Logging in...</div>;
  let xhr = new XMLHttpRequest();
  xhr.open('POST', '/login', true);
  xhr.setRequestHeader('Accept', 'application/json');
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.onreadystatechange = function() {
    if (xhr.readyState === XMLHttpRequest.DONE) {
      console.log('response', xhr.responseText);
      let {participant_id} = JSON.parse(xhr.responseText);
      window.location.replace(`${window.location.protocol}//${window.location.host}/?${participant_id}-p`);
    }
  };
  xhr.send(JSON.stringify({params}));

} else {
  let match = query.match(/^(\w+)-(\w+)$/);
  if (match) {
    let clientId = match[1];
    let clientKind = match[2];
    let mod = require('./App');
    let globalState = mod.init(clientId, clientKind);
    let App = mod.default;
    topLevel = <App global={globalState} />;
  } else {
    topLevel = <h3>Invalid URL</h3>;
  }

}


ReactDOM.render(
  topLevel,
  document.getElementById('root')
);
