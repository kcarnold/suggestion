import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import seedrandom from 'seedrandom';

function experimentBlock({block, prewriteTimer, editTimer}) {
  return [
    {preEvent: {type: 'setupExperiment', block}, controllerScreen: 'Instructions'},
    {screen: 'ExperimentScreen', timer: prewriteTimer},
    {preEvent: {type: 'setEditFromExperiment'}, screen: null, controllerScreen: 'EditScreen', timer: editTimer},
    {controllerScreen: 'PostTaskSurvey'},
  ];
}

const prewriteTimer = 120;
const editTimer = 120;


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
      timerStartedAt: null,
      timerDur: null,
      get screens() {
        return [
          {controllerScreen: 'Consent', screen: 'ProbablyWrongCode'},
          {screen: 'SetupPairingPhone', controllerScreen: 'SetupPairingComputer'},
          {controllerScreen: 'ConfirmPairing'},
          {controllerScreen: 'SelectRestaurants'},
          ...experimentBlock({block: 0, prewriteTimer, editTimer}),
          ...experimentBlock({block: 1, prewriteTimer, editTimer}),
          {controllerScreen: 'PostExpSurvey'},
          {screen: 'Done', controllerScreen: 'Done'},
        ];
      },
      get places() {
        let {controlledInputs} = this;
        let res = [
          {name: controlledInputs.get('restaurant1'), visit: controlledInputs.get('visit1'), stars: controlledInputs.get('star1')},
          {name: controlledInputs.get('restaurant2'), visit: controlledInputs.get('visit2'), stars: controlledInputs.get('star2')}
        ]
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
      get curPlace() {
        return this.places[this.block];
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
    case 'setScreen':
      this.screenNum = event.screen;
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
    case 'setTimer':
      this.timerStartedAt = event.start;
      this.timerDur = event.dur;
      break;
    default:
    }
  }
}
