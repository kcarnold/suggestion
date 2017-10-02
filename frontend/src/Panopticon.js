import React, { Component } from 'react';
import * as M from 'mobx';
import moment from 'moment';
import {observer, Provider} from 'mobx-react';
import WSClient from './wsclient';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './MasterView';
import _ from 'lodash';

let match = window.location.search.slice(1).match(/^(\w+)-(\w+)$/);
let panopt = match[1], panopticode = match[2];

const SHOW_REPLAY = false;

var ws = new WSClient(`ws://${window.location.host}/ws`);
ws.setHello([{type: 'init', participantId: panopticode, kind: panopt}]);
ws.connect();

// Logs are not observable, for minimal overhead.
var logs = {};

export class PanoptStore {
  constructor(clientId, kind) {
    M.extendObservable(this, {
      showingIds: [],
      states: M.observable.shallowMap({}),
      startTimes: M.observable.shallowMap({}),
      times: M.observable.shallowMap({}),
      acceleration: 10,
      analyses: M.observable.shallowMap()
    });
  }

  addViewer = M.action((id) => {
    if (this.showingIds.indexOf(id) !== -1) return; // Already a viewer.
    this.showingIds.push(id);
    if (!this.states.has(id)) {
      this.states.set(id, new MasterStateStore(id));
      if (SHOW_REPLAY) {
        ws.send({type: 'get_logs', participantId: id});
      }
      ws.send({type: 'get_analyzed', participantId: id});
    }
  });

  addViewers = M.action((ids) => {
    ids.split(/\s/).forEach(id => this.addViewer(id))
  })
}

var store = new PanoptStore();

function replay(log, state) {
  if (log.length === 0) return;
  let idx = 0;
  function tick() {
    let event = log[idx];
    let toLog = {...event};
    delete toLog.participant_id;
    delete toLog.timestamp;
    delete toLog.kind;
    delete toLog.jsTimestamp;
    if (event.type === 'requestSuggestions') {
      let requestCurText = event.request.sofar + event.request.cur_word.map((ent => ent.letter)).join('');
      if (event.request.request_id === state.experimentState.contextSequenceNum && requestCurText !== state.experimentState.curText) {
        debugger;
      }
    }
    state.handleEvent(event);
    if (idx === log.length - 1) return;
    setTimeout(tick, Math.min(1000, (log[idx + 1].jsTimestamp - log[idx].jsTimestamp) / store.acceleration));
    idx++;
  }
  tick();
}

ws.onmessage = function(msg) {
  if (msg.type === 'logs') {
    let participantId = msg.participant_id;
    logs[participantId] = msg.logs;
    let state = store.states.get(participantId);
    replay(msg.logs, state);
    state.replaying = false;
    // store.startTimes.set(participantId, msg.logs[0].jsTimestamp);
    // state.replaying = true;
    // logs[participantId].forEach(msg => {
    //   state.handleEvent(msg);
    // });
    // state.replaying = false;
  }
  if (msg.type === 'analyzed') {
    store.analyses.set(msg.participant_id, msg.analysis);
  }
};

const nullDispatch = () => {};

const ScreenTimesTable = ({screenTimes}) => {
  let lastTime = null;
  let durs = [];
  screenTimes.forEach(({timestamp}) => {
    let curTime = moment(timestamp);
    if (lastTime !== null) {
      durs.push(curTime.diff(lastTime, 'minutes', true));
    }
    lastTime = curTime;
  });
  return <table>
    <tbody>
      {screenTimes.map(({name, num, timestamp}, i) => {
        let curTime = moment(timestamp);
        let dur = i < durs.length ? `${Math.round(10 * durs[i]) / 10} min` : null;
        return <tr key={num}><td>{name}</td><td>{curTime.format('LTS')}</td><td>{dur}</td></tr>;
      })}
    </tbody>
  </table>;
}

const ShowRecs = ({recs, action}) => <div
  style={{display: 'flex', flexFlow: 'column nowrap'}}>
  <div style={{display: 'flex', flexFlow: 'row nowrap', justifyContent: 'space-between'}}>
    {recs.synonyms.map(({words}, i) => <div key={i} style={{padding: '5px', fontWeight: action.slot === i ? 'bold' : null}}>{words.join(' ')}</div>)}
  </div>
</div>;

function getStyle(chunk) {
  let {actionClass, action, chars} = chunk;
  let {sugInserted} = action;
  if (actionClass === 'tapKey') return {};
  let style = {};
  // let curWord = (action.curText.match(/\w+$/) || [''])[0];
  // console.assert(actionClass.match(/^tapSugg/));
  if (sugInserted !== chars.trim()) {
    style['background'] = 'red';
    console.log('mismatch "%s" vs "%s", context "%s" sug "%s"', sugInserted, chars, action.curText.slice(-15), action.visibleSuggestions[action.which]);
  } else if (actionClass === 'tapSugg_part') {
    style['outline'] = '1px solid red';
  } else {
    style['background'] = 'green';
  }
  return style;
}

const Chunk = ({chunk}) => <span style={getStyle(chunk)}>{chunk.chars}</span>;

const AnnotatedFinalText = ({chunks}) => <div className="AnnotatedFinalText">
  {chunks.map((chunk, i) => <Chunk key={i} chunk={chunk} />)}
</div>;

const AnalyzedView = observer(({store, participantId}) => {
  let analysis = store.analyses.get(participantId);
  if (!analysis) return null;
  return <div>
    {_.map(analysis.byExpPage, ((content, pageName) => {
      let synonymTaps = _.filter(content.displayedSuggs, {'action': {type: 'tapSuggestion', which: 'synonyms'}});
      return <div key={pageName}>
        {pageName} ({content.condition}) ({JSON.stringify(content.place)})
        <AnnotatedFinalText chunks={content.chunks} />
        <table>
          <tbody>
            {synonymTaps.map(({context, recs, action}, i) =>
              <tr key={i}>
                <td style={{maxWidth: '200px', fontSize: '10px', overflow: 'hidden', whiteSpace: 'nowrap', direction: 'rtl'}}>
                  <bdi>{context}</bdi>
                </td>
                <td><ShowRecs recs={recs} action={action} /></td>
              </tr>)}
          </tbody>
        </table>
      </div>;
    }))}
    <ScreenTimesTable screenTimes={analysis.screenTimes} />
    <table style={{fontSize: '10px'}}>
      <tbody>
        {Object.entries(analysis.allControlledInputs).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
      </tbody>
    </table>
  </div>;
});

const ReplayView = observer(({store, participantId}) => {
  let state = store.states.get(participantId);
  if (!state.masterConfig) return null;

  return <div style={{display: 'flex', flexFlow: 'row'}}>
      <div style={{overflow: 'hidden', width: state.phoneSize.width, height: state.phoneSize.height, border: '1px solid black', flex: '0 0 auto'}}>
        <Provider state={state} dispatch={nullDispatch} clientId={participantId} clientKind={'p'} spying={true}>
          <MasterView kind={'p'}/>
        </Provider>
      </div>
      <div style={{overflow: 'hidden', width: 500, height: 700, border: '1px solid black', flex: '0 0 auto'}}>
        <Provider state={state} dispatch={nullDispatch} clientId={participantId} clientKind={'c'} spying={true}>
          <MasterView kind={'c'} />
        </Provider>
      </div>
      <div style={{flex: '1 1 auto'}}>
        {state.experiments.entries().map(([name, expState]) => <div key={name}>
          <b>{name}</b><br/>{expState.curText}</div>)}
      </div>
    </div>;
});

const Panopticon = observer(class Panopticon extends Component {
  render() {
    return <div>
      <div><input ref={elt => {this.viewerInput = elt;}} /><button onClick={evt => {
        store.addViewers(this.viewerInput.value);
        this.viewerInput.value = '';
      }}>Add</button></div>
      {store.showingIds.map(participantId => {
        let conditions = []
        // let state = store.states.get(participantId);
        // if (!state.masterConfig) return null;
        return <div key={participantId}>
          <h1>{participantId} {conditions.join(',')}</h1>
          <AnalyzedView store={store} participantId={participantId} />
          {SHOW_REPLAY && <ReplayView store={store} participantId={participantId} />}
        </div>;
      })}</div>;
  }
});

export default Panopticon;

// Globals
window.M = M;
window.store = store;


// store.addViewers('c104c0 feebe1 50b0c9 50b80b 99c66d')
// store.addViewers('2wr5j9 v4w898 7jqggr')
// store.addViewers('9qxf5g hfj33r jcqf4w')
// store.addViewers('7jqggr');
// store.addViewers('w5hfrr 376j3q vcgfhq');
// store.addViewers('7x3v6q h72hhj')
store.addViewers(
  // '2vgwmf gfhfhx rwq22w'
  // 'rwq22w'
  // 'w5325r',
  'rwq22w 2vgwmf gfhfhx 9mvhv7 c8c75v w5325r 846ch3 g3h79x c97rm4 6cxf2g 7fvfhm w832c6'
  )

