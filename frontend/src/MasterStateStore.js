import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import seedrandom from 'seedrandom';


export class MasterStateStore {
  constructor(clientId) {
    this.__version__ = 1;
    this.clientId = clientId;
    this.rng = seedrandom(clientId);
    // Don't disturb the calling sequence of the rng, or state will become invalid.
    this.conditionOrder = ['wp', 'pw'][this.rng() < .5 ? 0 : 1];
    this.swapPlaceOrder = this.rng() < .5;

    M.extendObservable(this, {
      screenNum: 0,
      block: null,
      experimentState: null,
      controlledInputs: {},
      get places() {
        let {controlledInputs} = this;
        let res = [
          {name: controlledInputs.restaurant1, visit: controlledInputs.visit1},
          {name: controlledInputs.restaurant2, visit: controlledInputs.visit2}
        ];
        if (this.swapPlaceOrder) {
          res.unshift(res.pop());
        }
        return res;
      },
      get suggestionRequestParams() {
        return {
          rare_word_bonus: this.block === 0 ? 1 : 0.,
          domain: 'yelp_train'
        };
      }
    });
  }

  handleEvent = (event) => {
    if (this.experimentState) {
      this.experimentState.handleEvent(event);
    }
    switch (event.type) {
    case 'next':
      this.screenNum++;
      break;
    case 'setupExperiment':
      this.experimentState = new ExperimentStateStore();
      this.block = event.block;
      break;
    case 'controlledInputChanged':
      this.controlledInputs[event.name] = event.value;
      break;
    default:
    }
  }
}
