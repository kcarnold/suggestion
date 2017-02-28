import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import seedrandom from 'seedrandom';

function experimentBlock({block, prewriteTimer, editTimer}) {
  return [
    {preEvent: {type: 'setupExperiment', block}, controllerScreen: 'Instructions'},
    {screen: 'ExperimentScreen', timer: prewriteTimer},
    {screen: 'BreakBeforeEditPhone', controllerScreen: 'BreakBeforeEdit'},
    {preEvent: {type: 'setEditFromExperiment'}, screen: null, controllerScreen: 'EditScreen', timer: editTimer},
    {controllerScreen: 'PostTaskSurvey'},
  ];
}

const prewriteTimer = 60 * 3;
const editTimer = 60 * 2;

const ngramFlags = {
  useSufarr: false,
  temperature: 0,
};

const namedConditions = {
  word: {
    sugFlags: ngramFlags,
    showPhrase: false
  },
  phrase: {
    sugFlags: ngramFlags,
    showPhrase: true
  },
  rarePhrase: {
    sugFlags: {
      useSufarr: true,
      rare_word_bonus: 1,
    },
    showPhrase: true
  }
};

export class MasterStateStore {
  constructor(clientId, kind) {
    this.__version__ = 1;
    this.clientId = clientId;
    this.kind = kind;

    this.rng = seedrandom(clientId);
    // Don't disturb the calling sequence of the rng, or state will become invalid.
    this.swapConditionOrder = this.rng() < .5;
    this.swapPlaceOrder = this.rng() < .5;
    this.conditions = ['word', 'phrase'];
    if (this.swapConditionOrder) {
      this.conditions.unshift(this.conditions.pop());
    }

    let isDemo = (clientId || '').slice(0, 4) === 'demo';

    M.extendObservable(this, {
      replaying: true,
      screenNum: 0,
      block: null,
      experimentState: null,
      controlledInputs: M.asMap({}),
      timerStartedAt: null,
      timerDur: null,
      get screens() {
        if (isDemo) return [{screen: 'ExperimentScreen', controllerScreen: 'ExperimentScreen'}];
        return [
          {controllerScreen: 'Welcome', screen: 'ProbablyWrongCode'},
          {screen: 'SetupPairingPhone', controllerScreen: 'SetupPairingComputer'},
          {controllerScreen: 'ConfirmPairing'},
          {controllerScreen: 'SelectRestaurants'},
          ...experimentBlock({block: 0, prewriteTimer, editTimer}),
          ...experimentBlock({block: 1, prewriteTimer, editTimer}),
          {controllerScreen: 'PostExpSurvey'},
          {screen: 'Done', controllerScreen: 'Done'},
        ];
      },
      get nextScreen() {
        return this.screens[this.screenNum + 1];
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
          ...this.condition.sugFlags,
          domain: 'yelp_train'
        };
      },
      get curPlace() {
        if (isDemo) return {name: 'Corner Cafe', visit: 'last night', stars: 4};
        return this.places[this.block];
      },
      get curEditTextName() { return 'edited-'+this.block; },
      get curEditText() {
        return this.controlledInputs.get(this.curEditTextName);
      },
      get condition() {
        if (isDemo) return namedConditions[clientId.slice(4)];
        return namedConditions[this.conditions[this.block]];
      }
    });

    if (isDemo) {
      this.block = 0;
      this.experimentState = new ExperimentStateStore(this.condition);
    }
  }

  handleEvent = (event) => {
    if (this.experimentState) {
      this.experimentState.handleEvent(event);
    }
    switch (event.type) {
    case 'externalAction':
      if (event.externalAction === 'completeSurvey') {
        this.screenNum++;
      } else {
        alert("Unknown externalAction: "+event.externalAction);
      }
      break;
    case 'next':
      this.screenNum++;
      break;
    case 'setScreen':
      this.screenNum = event.screen;
      break;
    case 'setupExperiment':
      this.block = event.block;
      this.experimentState = new ExperimentStateStore(this.condition);
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
