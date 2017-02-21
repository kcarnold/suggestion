import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';


export class MasterStateStore {
  constructor() {
    this.__version__ = 1;
    M.extendObservable(this, {
      screenNum: 0,
      block: null,
      experimentState: null,
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
    default:
    }
  }
}
