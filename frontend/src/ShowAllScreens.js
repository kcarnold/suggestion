import React, { Component } from 'react';
import _ from 'lodash';
import M from 'mobx';
import {observer, Provider} from 'mobx-react';
import {MasterStateStore} from './MasterStateStore';
import {MasterView} from './Views';

let states = [new MasterStateStore('zzzzzz')];
let screens = states[0].screens;
states[0].replaying = false;
for (let i=1; i<screens.length; i++) {
  let newState = new MasterStateStore('zzzzzz');
  for (let j=0; j<i; j++) {
    newState.handleEvent({type: 'next'});
    if (newState.curScreen.screen === 'ExperimentScreen') {
      _.forEach(`${j}`, chr => newState.handleEvent({type: 'tapKey', key: chr}));
    }
  }
  states.push(newState);
  newState.replaying = false;
}

const ShowAllScreens = observer(class ShowAllScreens extends Component {
  render() {
    return <div>
      {states.map((state, i) => <div key={i} style={{display: 'flex', flewFlow: 'row'}}>
        <div style={{overflow: 'hidden', width: 360, height: 599, border: '1px solid black'}}>
          <Provider state={state} dispatch={() => {}} clientId={'zzzzzz'} clientKind={'p'} spying={true}>
            <MasterView kind={'p'}/>
          </Provider>
        </div>
        <div style={{overflow: 'hidden', width: 500, height: 700, border: '1px solid black'}}>
          <Provider state={state} dispatch={() => {}} clientId={'zzzzzz'} clientKind={'c'} spying={true}>
            <MasterView kind={'c'} />
          </Provider>
        </div>
      </div>)
    }</div>;
  }
});

export default ShowAllScreens;
