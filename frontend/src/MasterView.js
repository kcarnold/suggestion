import React from 'react';
import _ from 'lodash';
import {observer, inject} from 'mobx-react';
import * as Views from './Views';


export const MasterView = inject('state')(observer(({state, kind}) => {
  if (state.replaying) return <div>Loading...</div>;
  let screenDesc = state.screens[state.screenNum];
  let screenName;
  if (kind === 'c') {
    screenName = screenDesc.controllerScreen || 'LookAtPhone';
  } else {
    screenName = screenDesc.screen || 'LookAtComputer';
  }
  return (
    <div className="App">
      {React.createElement(Views[screenName])}
    </div>);
}));

export default MasterView;
