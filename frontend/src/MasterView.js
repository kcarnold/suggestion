import React, {Component} from 'react';
import {observer, inject} from 'mobx-react';
import * as Views from './Views';


export const MasterView = inject('state')(observer(class MasterView extends Component {
  componentDidUpdate() {
    window.scrollTo(0, 0);
  }

  render() {
    let {state, kind} = this.props;
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
  }
}));

export default MasterView;
