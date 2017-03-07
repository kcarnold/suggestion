import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import seedrandom from 'seedrandom';

const prewriteTimer = 60 * 5;
const finalTimer = 60 * 5;

class TutorialTasks {
  constructor() {
    M.extendObservable(this, {
      lastTapInSlot: {},
      consectutiveTaps: null,
      tasks: {
        tapSuggestion: false,
        doubleTap: false,
        tripleTap: false,
        typeKeyboard: false
      },
      get allDone() {
        let {tasks} = this;
        return _.every(tasks);
      }
    });
  }

  handleEvent(event) {
    switch(event.type) {
    case 'tapSuggestion':
      if (event.slot === 0) {
        this.tasks['tapSuggestion'] = true;
      }
      this.lastTapInSlot[event.slot] = event.jsTimestamp;
      if (this.consectutiveTaps && this.consectutiveTaps.slot === event.slot) {
        this.consectutiveTaps.times++;
        if (this.consectutiveTaps.times === 2 && event.slot === 1) {
          this.tasks.doubleTap = true;
        } else if (this.consectutiveTaps.times === 3 && event.slot === 2) {
          this.tasks.tripleTap = true;
        }
      } else {
        this.consectutiveTaps = {slot: event.slot, times: 1};
      }
      break;
    case 'tapKey':
      this.tasks.typeKeyboard = true;
      break;
    default:
    }
  }
}

function experimentBlock({block}) {
  return [
    {preEvent: {type: 'setupExperiment', block}, controllerScreen: 'Instructions'},
    {screen: 'ExperimentScreen', timer: prewriteTimer, isPrewrite: true},
    {preEvent: {type: 'setupExperiment', block}, screen: 'BreakBeforeEditPhone', controllerScreen: 'BreakBeforeEdit'},
    // {preEvent: {type: 'setEditFromExperiment'}, screen: null, controllerScreen: 'EditScreen', timer: editTimer},
    {screen: 'ExperimentScreen', controllerScreen: 'RevisionComputer', timer: finalTimer, isPrewrite: false},
    {controllerScreen: 'PostTaskSurvey'},
  ];
}

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
  constructor(clientId) {
    this.__version__ = 1;
    this.clientId = clientId;

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
      lastEventTimestamp: null,
      replaying: true,
      screenNum: 0,
      block: null,
      experiments: [],
      get experimentState() {
        if (this.experiments.length) {
          return this.experiments[this.experiments.length - 1];
        }
      },
      get prevExperimentState() {
        if (this.experiments.length > 1) {
          return this.experiments[this.experiments.length - 2];
        }
      },
      controlledInputs: M.asMap({}),
      timerStartedAt: null,
      timerDur: null,
      tutorialTasks: new TutorialTasks(),
      get screens() {
        if (isDemo) return [{screen: 'ExperimentScreen', controllerScreen: 'ExperimentScreen'}];
        return [
          {controllerScreen: 'Welcome', screen: 'ProbablyWrongCode'},
          {screen: 'SetupPairingPhone', controllerScreen: 'SetupPairingComputer'},
          {preEvent: {type: 'setupExperiment', block: 0}, screen: 'PracticePhone', controllerScreen: 'PracticeComputer'},
          {preEvent: {type: 'setupExperiment', block: 1}, screen: 'PracticePhone', controllerScreen: 'PracticeComputer2'},
          {controllerScreen: 'SelectRestaurants'},
          ...experimentBlock({block: 0}),
          ...experimentBlock({block: 1}),
          {controllerScreen: 'PostExpSurvey'},
          {screen: 'Done', controllerScreen: 'Done'},
        ];
      },
      get curScreen() {
        return this.screens[this.screenNum];
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
      get conditionName() {
        if (isDemo) return clientId.slice(4);
        return this.conditions[this.block];
      },
      get condition() {
        return namedConditions[this.conditionName];
      },
      get suggestionRequest() {
        let {experimentState} = this;
        if (!experimentState)
          return null;

        let seqNum = experimentState.contextSequenceNum;
        let {prefix, curWord} = experimentState.getSuggestionContext();
        return {
          type: 'requestSuggestions',
          request_id: seqNum,
          sofar: prefix,
          cur_word: curWord,
          ...this.suggestionRequestParams
        };
      }
    });

    if (isDemo) {
      this.block = 0;
      this.experiments.push(new ExperimentStateStore(this.condition));
    }
  }

  handleEvent = M.action((event) => {
    this.lastEventTimestamp = event.jsTimestamp;
    if (this.experimentState) {
      this.experimentState.handleEvent(event);
    }
    this.tutorialTasks.handleEvent(event);

    let screenAtStart = this.screenNum;
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
    case 'controlledInputChanged':
      this.controlledInputs.set(event.name, event.value);
      break;
    default:
    }

    if (this.screenNum !== screenAtStart) {
      // Execute start-of-screen actions.
      let screen = this.screens[this.screenNum];
      switch ((screen.preEvent || {}).type) {
      case 'setupExperiment':
        this.block = screen.preEvent.block;
        if (this.experimentState) {
          this.experimentState.dispose();
        }
        this.experiments.push(new ExperimentStateStore(this.condition));
        break;
      case 'setEditFromExperiment':
        this.controlledInputs.set(this.curEditTextName, this.experimentState.curText);
        break;
      default:
      }
      if (screen.timer) {
        this.timerStartedAt = event.jsTimestamp;
        this.timerDur = screen.timer;
      }
    }
  });
}
