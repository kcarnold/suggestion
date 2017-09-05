import React, { Component } from 'react';
import _ from 'lodash';
import {observer, Provider} from 'mobx-react';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './MasterView';

const fakeClientId = 'zzzzzz';

let externalAction = window.location.hash.slice(1);

let states = [];
let eventsSoFar = [];

function doEventToLastState(evt) {
  eventsSoFar.push(evt);
  states[states.length - 1].handleEvent(evt);
}
function copyState() {
  let newState = new MasterStateStore(fakeClientId);
  states.push(newState);
  eventsSoFar.forEach(evt => newState.handleEvent(evt));
  return newState;
}

if (window.location.search.slice(1) === 'showall') {
  copyState();
  doEventToLastState({type: 'externalAction', externalAction});
  let screens = states[0].screens;
  states[0].replaying = false;
  for (let i=1; i<screens.length; i++) {
    let newState = copyState();
    doEventToLastState({type: 'next'});
    if (newState.curScreen.screen === 'ExperimentScreen') {
      _.forEach(`${i}`, chr => {
        doEventToLastState({type: 'tapKey', key: chr});
      });
    }
    newState.replaying = false;
  }
}

const ShowAllScreens = observer(class ShowAllScreens extends Component {
  render() {
    function innerView(i, state, kind) {
      return <Provider
        state={state}
        dispatch={event => {
          event.jsTimestamp = +new Date();
          event.kind = kind;
          state.handleEvent(event);
        }}
        clientId={fakeClientId}
        clientKind={kind}
        spying={true}
      ><MasterView kind={kind}/></Provider>;
    }
    return <div>
      {states.map((state, i) => <div key={i} style={{display: 'flex', flewFlow: 'row'}}>
        <div style={{overflow: 'hidden', width: 360, height: 599, border: '1px solid black'}}>
          {innerView(i, state, 'p')}
        </div>
        <div style={{overflow: 'hidden', width: 500, height: 700, border: '1px solid black'}}>
          {innerView(i, state, 'c')}
        </div>
      </div>)
    }</div>;
  }
});

export default ShowAllScreens;
