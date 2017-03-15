// @flow

import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import seedrandom from 'seedrandom';

const prewriteTimer = 60 * 5;
const finalTimer = 60 * 5;

let multiTapThresholdMs = 500;

class TutorialTasks {
  tasks: {[name: string]: boolean} ;
  consectutiveTaps: Object;

  constructor() {
    M.extendObservable(this, {
      consectutiveTaps: {},
      tasks: {
        tapSuggestion: false,
        doubleTap: false,
        quadTap: false,
        typeKeyboard: false
      },
      get allDone() {
        let {tasks} = this;
        return _.every(tasks);
      }
    });
  }

  handleEvent(event) {
    let timestamp = event.jsTimestamp;
    switch(event.type) {
    case 'tapSuggestion':
      this.tasks['tapSuggestion'] = true;
      if (this.consectutiveTaps.slot === event.slot && timestamp - this.consectutiveTaps.lastTimestamp < multiTapThresholdMs) {
        this.consectutiveTaps.times++;
        this.consectutiveTaps.lastTimestamp = timestamp;
        if (this.consectutiveTaps.times >= 2) {
          this.tasks.doubleTap = true;
        }
        if (this.consectutiveTaps.times >= 4) {
          this.tasks.quadTap = true;
        }
      } else {
        this.consectutiveTaps = {slot: event.slot, times: 1, lastTimestamp: event.jsTimestamp};
      }
      break;
    case 'tapKey':
      this.tasks.typeKeyboard = true;
      this.consectutiveTaps = {};
      break;
    case 'tapBackspace':
      this.consectutiveTaps = {};
      break;
    default:
    }
  }
}

type Screen = {
  controllerScreen: string,
  screen: string,
  preEvent?: Object
};

function experimentBlock({block}:{block: number}): Array<Screen> {

  return [
    {
      controllerScreen: 'Instructions', screen: 'ReadyPhone',
      preEvent: {type: 'setupExperiment', block, name: `pre-${block}`},
    },
    {screen: 'ExperimentScreen', controllerScreen: 'Instructions', timer: prewriteTimer},
    {screen: 'TimesUpPhone', controllerScreen: 'PostFreewriteSurvey'},
    {preEvent: {type: 'setupExperiment', block, name: `final-${block}`}, controllerScreen: 'Instructions', screen: 'ReadyPhone'},
    // {preEvent: {type: 'setEditFromExperiment'}, screen: null, controllerScreen: 'EditScreen', timer: editTimer},
    {screen: 'ExperimentScreen', controllerScreen: 'RevisionComputer', timer: finalTimer},
    {screen: 'TimesUpPhone', controllerScreen: 'PostTaskSurvey'},
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
  clientId: string;
  swapConditionOrder: boolean;
  swapPlaceOrder: boolean;
  conditions: Array<string>;
  lastEventTimestamp: number;
  replaying: boolean;
  block: number;
  times: {[name: string]: number};
  screenTimes: Array<{num: number, timestamp: number}>;
  screens: Array<Screen>;
  screenNum: number;
  curExperiment: string;
  condition: string;

  constructor(clientId:string) {
    this.clientId = clientId;

    let rng = seedrandom(clientId);
    // Don't disturb the calling sequence of the rng, or state will become invalid.
    this.swapConditionOrder = rng() < .5;
    this.swapPlaceOrder = rng() < .5;
    this.conditions = ['rarePhrase', 'phrase'];
    if (this.swapConditionOrder) {
      this.conditions.unshift(this.conditions.pop());
    }

    let isDemo = (clientId || '').slice(0, 4) === 'demo';

    this.times = {prewriteTimer, finalTimer};

    M.extendObservable(this, {
      lastEventTimestamp: null,
      replaying: true,
      screenNum: 0,
      block: null,
      experiments: M.asMap({}),
      curExperiment: null,
      get experimentState() {
        if (this.curExperiment) {
          return this.experiments.get(this.curExperiment);
        }
      },
      get isPrewrite() {
        return this.curExperiment.slice(0, 3) === 'pre';
      },
      controlledInputs: M.asMap({}),
      timerStartedAt: null,
      timerDur: null,
      tutorialTasks: new TutorialTasks(),
      screenTimes: [],
      passedQuiz: false,
      phoneSize: {width: 360, height: 500},
      get blockName() {
        switch (this.block) {
        case 0:
          return 'A';
        case 1:
          return 'B';
        default:
          return null;
        }
      },
      get screens() {
        if (isDemo) return [{preEvent: {type: 'setupExperiment', block: 0, name: 'demo'}, screen: 'ExperimentScreen', controllerScreen: 'ExperimentScreen'}];
        return [
          {controllerScreen: 'Welcome', screen: 'ProbablyWrongCode'},
          {screen: 'SetupPairingPhone', controllerScreen: 'SetupPairingComputer'},
          {controllerScreen: 'IntroSurvey'},
          {preEvent: {type: 'setupExperiment', block: 0, name: 'practice-0'}, screen: 'PracticePhone', controllerScreen: 'PracticeComputer'},
          {controllerScreen: 'SelectRestaurants'},
          ...experimentBlock({block: 0}),
          {preEvent: {type: 'setupExperiment', block: 1, name: 'practice-1'}, screen: 'PracticePhone', controllerScreen: 'PracticeComputer2'},
          ...experimentBlock({block: 1}),
          {screen: 'ShowReviews', controllerScreen: 'PostExpSurvey'},
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
    this.initScreen();
  }

  initScreen() {
    // Execute start-of-screen actions.
    let screen = this.screens[this.screenNum];
    if (screen.preEvent) {
      let event = screen.preEvent;
      switch (event.type) {
      case 'setupExperiment':
        this.block = screen.preEvent.block;
        if (this.experimentState) {
          this.experimentState.dispose();
        }
        this.curExperiment = event.name;
        this.experiments.set(event.name, new ExperimentStateStore(this.condition));
        break;
      case 'setEditFromExperiment':
        this.controlledInputs.set(this.curEditTextName, this.experimentState.curText);
        break;
      default:
      }
    }
    this.screenTimes.push({num: this.screenNum, timestamp: this.lastEventTimestamp});
    if (screen.timer) {
      this.timerStartedAt = this.lastEventTimestamp;
      this.timerDur = screen.timer;
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
      } else if (event.externalAction === 'passedQuiz') {
        this.passedQuiz = true;
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
    case 'setStateVarMagic':
      this[event.var] = event.val;
      break;
    case 'controlledInputChanged':
      this.controlledInputs.set(event.name, event.value);
      break;
    case 'resized':
      if (event.kind === 'p') {
        this.phoneSize = {width: event.width, height: event.height};
      }
      break;
    default:
    }

    if (this.screenNum !== screenAtStart) {
      this.initScreen();
    }
  });
}
