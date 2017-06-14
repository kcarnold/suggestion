import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
import moment from 'moment';
import {observer, Provider} from 'mobx-react';
import WSClient from './wsclient';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './MasterView';


class Stats {
  constructor() {
    M.extendObservable(this, {
      numInflight: 0,
      nosugg: [],
      diverse: [],
      match: [],
      get nosugg_mean() { return _.mean(this.nosugg); },
      get diverse_mean() { return _.mean(this.diverse); },
      get match_mean() { return _.mean(this.match); },
      get means() {
        return {
          nosugg: this.nosugg_mean,
          diverse: this.diverse_mean,
          match: this.match_mean
        };
      },
    });
  }
};

let stats = new Stats();
window.stats = stats;

const Requester = (requests) => {
  var ws = new WSClient(`ws://${window.location.host}/ws`);
  ws.setHello([{type: 'init', participantId: 'demobench', kind: 'bench'}]);

  let responses = [];
  let newRequestTimestamps = [];

  ws.onmessage = msg => {
    if (msg.type === 'suggestions') {
      let idx = responses.length;
      if (msg.timestamp !== newRequestTimestamps[idx]) debugger;
      let request = requests[idx];
      msg.timestamp = +new Date();
      responses.push(msg);
      stats.numInflight--;
      let rtt = msg.timestamp - newRequestTimestamps[idx];
      let sentType = request.sentiment;
      if (!sentType) sentType = 'nosugg';
      else if (sentType !== 'diverse') sentType = 'match';
      stats[sentType].push(rtt);
      let suggs = msg.next_word.map(s => (s.words || []).concat(s.continuation[0].words).join(' '));
      console.log(request.cur_word.length, suggs.join('; '));
      if (request.cur_word.length === 0 && suggs.filter(x=>x.length).length === 0) debugger
    }
  };

  let curTimeout = null;

  ws.stateChanged = newState => {
    if (newState === 'open') {
      if (curTimeout === null)
        setTimeout(sendNext, 2000*Math.random());
    }
  };

  function sendNext() {
    curTimeout = null;
    if (newRequestTimestamps.length >= requests.length) return;
    let idx = newRequestTimestamps.length;
    let {constraints, sofar, cur_word, sentiment, domain, request_id, temperature, useSufarr, use_bos_suggs, type} = requests[idx];
    let {timestamp} = ws.send({constraints, sofar, cur_word, sentiment, domain, request_id, temperature, useSufarr, use_bos_suggs, type});
    newRequestTimestamps.push(timestamp);
    stats.numInflight++;

    if (idx < requests.length - 1) {
      // Schedule another.
      let nextTs = requests[idx + 1].timestamp;
      let curTs = requests[idx].timestamp;
      // let diff = Math.min(1000, nextTs - curTs);
      let diff = 5000 * (stats.numInflight - 1) * Math.random();
      curTimeout = setTimeout(sendNext, diff);
    }
  }
  ws.connect();
}

function handleFiles(files) {
  var file = files[0];
  var reader = new FileReader();
  reader.onload = evt => {
    console.log("got data");
    var data = JSON.parse(reader.result);
    gotData(data);
  };
  reader.readAsText(file);
}

function gotData(data) {
  window.AllLogs = data;
  _.forEach(data, (requests, participant) => {
    Requester(_.filter(requests, x=>x.domain !== 'sotu'));
  });
}

const Bench = observer(class Bench extends Component {
  render() {
    return <div>
      Select the data file: <input type="file" onChange={evt => handleFiles(evt.target.files)} />

      <div>{stats.numInflight}</div>

      <div>{JSON.stringify(stats.means)}</div>
    </div>;
  }
})

export default Bench;

window.Requester = Requester;
