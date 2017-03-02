import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
import {observer, Provider} from 'mobx-react';
import WSClient from './wsclient';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './Views';

let match = window.location.search.slice(1).match(/^(\w+)-(\w+)$/);
let panopt = match[1], panopticode = match[2];

var ws = new WSClient(`ws://${window.location.host}/ws`);
ws.setHello([{type: 'init', participantId: panopticode, kind: panopt}]);
ws.connect();

// Logs are not observable, for minimal overhead.
var logs = {};

export class PanoptStore {
  constructor(clientId, kind) {
    M.extendObservable(this, {
      showingIds: [],
      states: M.asMap({}),
      startTimes: M.asMap({}),
      times: M.asMap({}),
    });
  }

  addViewer = M.action((id) => {
    if (this.showingIds.indexOf(id) !== -1) return; // Already a viewer.
    this.showingIds.push(id);
    if (!this.states.has(id)) {
      this.states.set(id, new MasterStateStore(id));
      ws.send({type: 'get_logs', participantId: id});
    }
  });
}

var store = new PanoptStore();

function replay(log, state) {
  if (log.length === 0) return;
  let idx = 0;
  function tick() {
    state.handleEvent(log[idx]);
    if (idx === log.length - 1) return;
    setTimeout(tick, Math.min(1000, (log[idx + 1].jsTimestamp - log[idx].jsTimestamp) / 10));
    idx++;
  }
  tick();
}

ws.onmessage = function(msg) {
  if (msg.type === 'logs') {
    let participantId = msg.participant_id;
    logs[participantId] = msg.logs;
    let state = new MasterStateStore(participantId);
    replay(msg.logs, state);
    state.replaying = false;
    // store.startTimes.set(participantId, msg.logs[0].jsTimestamp);
    // state.replaying = true;
    // logs[participantId].forEach(msg => {
    //   state.handleEvent(msg);
    // });
    // state.replaying = false;
    store.states.set(participantId, state);
  }
};



const Panopticon = observer(class Panopticon extends Component {
  render() {
    return <div>{store.showingIds.map(participantId => <div key={participantId}>
        <h1>{participantId}</h1>
        <div style={{overflow: 'hidden', width: 360, height: 500, border: '1px solid black'}}>
          <Provider state={store.states.get(participantId)} dispatch={() => {}} clientId={participantId} clientKind={'p'} spying={true}>
            <MasterView kind={'p'}/>
          </Provider>
        </div>
        <div style={{overflow: 'hidden', width: 360, height: 500, border: '1px solid black'}}>
          <Provider state={store.states.get(participantId)} dispatch={() => {}} clientId={participantId} clientKind={'c'} spying={true}>
            <MasterView kind={'c'} />
          </Provider>
        </div>
      </div>)
    }</div>;
  }
});

export default Panopticon;

// Globals
window.M = M;
window.store = store;