import React, { Component } from 'react';
import _ from 'lodash';
import {observer, Provider} from 'mobx-react';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './Views';

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
    if (newState.curScreen.screen === 'ReadyPhone' && !newState.passedQuiz) {
      newState.replaying = false;
      newState = copyState();
      doEventToLastState({type: 'externalAction', externalAction: 'passedQuiz'});
    }
    newState.replaying = false;
  }
}
const ShowAllScreens = observer(class ShowAllScreens extends Component {
  render() {
    return <div>
      {states.map((state, i) => <div key={i} style={{display: 'flex', flewFlow: 'row'}}>
        <div style={{overflow: 'hidden', width: 360, height: 599, border: '1px solid black'}}>
          <Provider state={state} dispatch={() => {}} clientId={fakeClientId} clientKind={'p'} spying={true}>
            <MasterView kind={'p'}/>
          </Provider>
        </div>
        <div style={{overflow: 'hidden', width: 500, height: 700, border: '1px solid black'}}>
          <Provider state={state} dispatch={() => {}} clientId={fakeClientId} clientKind={'c'} spying={true}>
            <MasterView kind={'c'} />
          </Provider>
        </div>
      </div>)
    }</div>;
  }
});

export default ShowAllScreens;
