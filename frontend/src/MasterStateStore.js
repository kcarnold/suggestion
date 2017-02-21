import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';


let START_PAGE = 'experiment';


export class MasterStateStore {
  constructor() {
    this.__version__ = 1;
    M.extendObservable(this, {
      block: 0,
      page: START_PAGE,
      experimentState: new ExperimentStateStore(),
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
    case 'typingDone':
      this.page = 'edit';
      break;
    case 'editingDone':
      if (this.block === 0) {
        this.block = 1;
        this.experimentState = new ExperimentStateStore();
        this.page = 'experiment';
      } else {
        this.page = 'postSurvey';
      }
      break;
    default:
    }
  }
}
