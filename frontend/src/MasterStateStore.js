import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import seedrandom from 'seedrandom';


export class MasterStateStore {
  constructor(clientId, kind) {
    this.__version__ = 1;
    this.clientId = clientId;
    this.kind = kind;
    this.rng = seedrandom(clientId);
    // Don't disturb the calling sequence of the rng, or state will become invalid.
    this.conditionOrder = ['wp', 'pw'][this.rng() < .5 ? 0 : 1];
    this.swapPlaceOrder = this.rng() < .5;

    M.extendObservable(this, {
      screenNum: 0,
      block: null,
      experimentState: null,
      controlledInputs: M.asMap({}),
      get places() {
        let {controlledInputs} = this;
        let res = [
          {name: controlledInputs.get('restaurant1'), visit: controlledInputs.get('visit1')},
          {name: controlledInputs.get('restaurant2'), visit: controlledInputs.get('visit2')}
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
      },
      get curEditTextName() { return 'edited-'+this.block; },
      get curEditText() {
        return this.controlledInputs.get(this.curEditTextName);
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
      this.controlledInputs.set(event.name, event.value);
      break;
    case 'setEditFromExperiment':
      this.controlledInputs.set(this.curEditTextName, this.experimentState.curText);
      break;
    default:
    }
  }
}
