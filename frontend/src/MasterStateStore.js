// @flow

import M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import TutorialTasks from './TutorialTasks';
import seedrandom from 'seedrandom';

const prewriteTimer = 60 * 5;
const finalTimer = 60 * 5;

type Screen = {
  controllerScreen: string,
  screen: string,
  preEvent?: Object,
  timer?: number,
};

function experimentBlock(block:number, conditionName: string, includePrewrite: boolean): Array<Screen> {
  let prewritePhase = [
    {
      controllerScreen: 'Instructions', screen: 'ReadyPhone',
      preEvent: {type: 'setupExperiment', block, condition: conditionName, name: `pre-${block}`},
    },
    {screen: 'ExperimentScreen', controllerScreen: 'Instructions', timer: prewriteTimer},
    {controllerScreen: 'PostFreewriteSurvey'},
  ];

  let finalPhase = [
    {preEvent: {type: 'setupExperiment', block, condition: conditionName, name: `final-${block}`}, controllerScreen: 'Instructions', screen: 'ReadyPhone'},
    // {preEvent: {type: 'setEditFromExperiment'}, screen: null, controllerScreen: 'EditScreen', timer: editTimer},
    {screen: 'ExperimentScreen', controllerScreen: 'RevisionComputer', timer: finalTimer},
    {controllerScreen: 'PostTaskSurvey'},
  ];

  if (includePrewrite) {
    return prewritePhase.concat(finalPhase);
  } else {
    return finalPhase;
  }
}

const ngramFlags = {
  useSufarr: false,
  temperature: 0,
  use_bos_suggs: false,
};

const namedConditions = {
  trump: {
    sugFlags: {
      useSufarr: false,
      temperature: 0,
      use_bos_suggs: false,
      domain: 'tweeterinchief'
    },
    showPhrase: true,
  },
  word: {
    sugFlags: ngramFlags,
    showPhrase: false
  },
  phrase: {
    //sugFlags: {...ngramFlags, continuation_length: 17},
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0,
      null_logprob_weight: 0,
      use_bos_suggs: false,
      continuation_length: 17,
    },
    showPhrase: true
  },
  rarePhrase: {
    sugFlags: {
      useSufarr: true,
      rare_word_bonus: 0,
      null_logprob_weight: -.75,
      use_bos_suggs: false,
      continuation_length: 17,
    },
    showPhrase: true
  },
  diverse: {
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: true,
      continuation_length: 17,
    },
    showPhrase: true
  },
  wdiverse: {
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: true,
      continuation_length: 17,
    },
    showPhrase: false
  },
  antidiverse: {
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: 'antidiverse',
      continuation_length: 17,
    },
    showPhrase: true
  },
  nondiverse: {
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: false,
      continuation_length: 17,
    },
    showPhrase: true
  },
  continue: {
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: 'continue',
      continuation_length: 17,
    },
    showPhrase: true
  },
  withPrewrite: {
    sugFlags: {
      useSufarr: true,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: false,
      continuation_length: 17,
    },
    showPhrase: true,
    usePrewriteText: true,
  }
};

const MASTER_CONFIGS = {
  demo: {
    baseConditions: ['word', 'phrase'],
  },
  study1: {
    baseConditions: ['word', 'phrase'],
    prewrite: false,
  },
  study2: {
    baseConditions: ['rarePhrase', 'phrase'],
    prewrite: true,
    instructions: 'detailed',
  },
  funny: {
    baseConditions: ['rarePhrase', 'phrase'],
    prewrite: true,
    instructions: 'funny',
  },
  study4: {
    baseConditions: ['rarePhrase', 'phrase'],
    prewrite: false,
    instructions: 'review',
  },
  diversity: {
    baseConditions: ['diverse', 'continue'],
    prewrite: false,
    instructions: 'review'
  },
  wdiversity: {
    baseConditions: ['diverse', 'wdiverse'],
    prewrite: false,
    instructions: 'review'
  },
  infoSource: {
    baseConditions: ['withPrewrite', 'phrase'],
    prewrite: false,
    instructions: 'review'
  }
};



function getScreens(masterConfigName: string, conditions: string[]) {
  let masterConfig = MASTER_CONFIGS[masterConfigName];
  let [c1, c2] = conditions;
  let result = [
    {controllerScreen: 'Welcome', screen: 'ProbablyWrongCode'},
    {screen: 'SetupPairingPhone', controllerScreen: 'SetupPairingComputer'},
    {preEvent: {type: 'setupExperiment', block: 0, condition: 'trump', name: 'practice'}, screen: 'PracticePhone', controllerScreen: 'PracticeComputer'},
    {controllerScreen: 'SelectRestaurants', screen: 'ProbablyWrongCode'},
    {controllerScreen: 'IntroSurvey'},
  ];
  if (masterConfigName === 'infoSource') {
    conditions.forEach((conditionName, block) => {
      result = result.concat([
        {preEvent: {type: 'setupExperiment', block, condition: conditionName, name: `final-${block}`}, controllerScreen: 'ListWords'},
        {screen: 'ExperimentScreen', controllerScreen: 'RevisionComputer', timer: finalTimer},
        {controllerScreen: 'PostTaskSurvey'},
      ]);
    });

  } else {
    result = result.concat([
      ...experimentBlock(0, conditions[0], masterConfig.prewrite),
      ...experimentBlock(1, conditions[1], masterConfig.prewrite),
    ]);
  }
  result = result.concat([
    {screen: 'ShowReviews', controllerScreen: 'PostExpSurvey'},
    {screen: 'Done', controllerScreen: 'Done'},
  ]);
  return result;
}



export class MasterStateStore {
  masterConfig: Object;
  masterConfigName: string;
  prewrite: boolean;
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
  timerDur: number;
  timerStartedAt: number;

  setMasterConfig(configName:string) {
    this.masterConfigName = configName;
    this.masterConfig = MASTER_CONFIGS[configName];
    let conditions = this.masterConfig.baseConditions.slice();
    if (this.swapConditionOrder) {
      conditions.unshift(conditions.pop());
    }
    this.conditions = conditions;
    this.initScreen();
  }

  constructor(clientId:string) {
    this.clientId = clientId;

    let rng = seedrandom(clientId);
    // Don't disturb the calling sequence of the rng, or state will become invalid.
    this.swapConditionOrder = rng() < .5;
    this.swapPlaceOrder = rng() < .5;

    let isDemo = (clientId || '').slice(0, 4) === 'demo';
    let demoConditionName = clientId.slice(4);

    this.times = {prewriteTimer, finalTimer};

    M.extendObservable(this, {
      masterConfig: null,
      participantCode: null,
      get prewrite() { return this.masterConfig.prewrite; },
      prewriteText: '',
      curPrewriteLine: 0,
      get prewriteLines() {
        let text = this.prewriteText.trim();
        if (text === '') return [];
        return text.split('\n');
      },
      lastEventTimestamp: null,
      replaying: true,
      screenNum: 0,
      block: null,
      conditions: null,
      conditionName: null,
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
      tutorialTasks: null,
      screenTimes: [],
      passedQuiz: false,
      phoneSize: {width: 360, height: 500},
      pingTime: null,
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
        if (isDemo) {
          return [{
            preEvent: {type: 'setupExperiment', block: 0, condition: demoConditionName, name: 'demo'},
            screen: 'ExperimentScreen', controllerScreen: 'ExperimentScreen'
          }];
        }
        return getScreens(this.masterConfigName, this.conditions);
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
          {name: controlledInputs.get('restaurant1'), visit: controlledInputs.get('visit1'), stars: controlledInputs.get('star1'), knowWhatToWrite: controlledInputs.get('knowWhat1')},
          {name: controlledInputs.get('restaurant2'), visit: controlledInputs.get('visit2'), stars: controlledInputs.get('star2'), knowWhatToWrite: controlledInputs.get('knowWhat2')}
        ]
        if (this.swapPlaceOrder) {
          res.unshift(res.pop());
        }
        return res;
      },
      get suggestionRequestParams() {
        return {
          domain: 'yelp_train',
          ...this.condition.sugFlags,
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
        console.assert(!!this.conditionName);
        return namedConditions[this.conditionName];
      },
      get suggestionRequest() {
        let {experimentState} = this;
        if (!experimentState)
          return null;

        let seqNum = experimentState.contextSequenceNum;
        let {prefix, curWord, constraints, promise} = experimentState.getSuggestionContext();
        let response = {
          type: 'requestSuggestions',
          request_id: seqNum,
          sofar: prefix,
          cur_word: curWord,
          constraints,
          promise,
          ...this.suggestionRequestParams
        };
        if (this.condition.usePrewriteText && this.prewriteLines.length) {
          response['prewrite_info'] = {
            text: this.prewriteLines[this.curPrewriteLine],
            amount: 1.
          };
        }
        return response;
      }
    });

    if (isDemo) {
      this.setMasterConfig('demo');
      if (demoConditionName === 'withPrewrite') {
        this.prewriteText = "with friend\nsat outside\nwait for server\nwater ran out"
      }
    }
  }

  initScreen() {
    // Execute start-of-screen actions.
    let screen = this.screens[this.screenNum];
    if (screen.preEvent) {
      let {preEvent} = screen;
      switch (preEvent.type) {
      case 'setupExperiment':
        this.block = preEvent.block;
        if (this.experimentState) {
          this.experimentState.dispose();
        }
        this.conditionName = preEvent.condition;
        this.curExperiment = preEvent.name;
        this.experiments.set(preEvent.name, new ExperimentStateStore(this.condition));
        this.tutorialTasks = new TutorialTasks();
        if (this.masterConfig.useConstraints) {
          this.experimentState.useConstraints = this.masterConfig.useConstraints;
        }
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
    if (this.tutorialTasks) {
      this.tutorialTasks.handleEvent(event);
    }

    let screenAtStart = this.screenNum;
    switch (event.type) {
    case 'externalAction':
      event.externalAction.split('&').forEach(action => {
        if (action.slice(0, 2) === 'c=') {
          this.setMasterConfig(action.slice(2));
        } else if (action.slice(0, 2) === 'p=') {
          this.participantCode = action.slice(2);
        } else if (action === 'completeSurvey') {
          this.screenNum++;
        } else if (action === 'passedQuiz') {
          this.passedQuiz = true;
        } else {
          alert("Unknown externalAction: "+action);
        }
      });
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
    case 'prewriteTextChanged':
      this.prewriteText = event.value;
      break;
    case 'addPrewriteItem':
      this.prewriteText = this.prewriteText.trim() + '\n' + event.line;
      break;
    case 'selectOutline':
      this.curPrewriteLine = event.idx;
      break;
    case 'resized':
      if (event.kind === 'p') {
        this.phoneSize = {width: event.width, height: event.height};
      }
      break;
    case 'pingResults':
      if (event.kind === 'p') {
        this.pingTime = event.ping.mean;
      }
      break;
    default:
    }

    if (this.screenNum !== screenAtStart) {
      this.initScreen();
    }
  });
}
