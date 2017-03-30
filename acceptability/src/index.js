import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

// import 'bootstrap/dist/css/bootstrap.css';
// import 'bootstrap/dist/css/bootstrap-theme.css';

let data = JSON.parse(document.getElementById('data').textContent);

ReactDOM.render(
  <App data={data} />,
  document.getElementById('root')
);

