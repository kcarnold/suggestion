// @flow

import * as M from 'mobx';
import _ from 'lodash';
import {ExperimentStateStore} from './ExperimentState';
import TutorialTasks from './TutorialTasks';
import {seededShuffle} from './shuffle';

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
      screen: 'ReadyPhone',
      preEvent: {type: 'setupExperiment', block, condition: conditionName, name: `pre-${block}`},
    },
    {screen: 'ExperimentScreen', controllerScreen: 'Instructions', timer: prewriteTimer},
    {controllerScreen: 'PostFreewriteSurvey'},
  ];

  let finalPhase = [
    {preEvent: {type: 'setupExperiment', block, condition: conditionName, name: `final-${block}`}, controllerScreen: 'Instructions', screen: 'ReadyPhone'},
    // {preEvent: {type: 'setEditFromExperiment'}, screen: null, controllerScreen: 'EditScreen', timer: editTimer},
    {screen: 'ExperimentScreen', controllerScreen: 'RevisionComputer', timer: finalTimer},
    {screen: 'PostTaskSurvey'},
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

const CONDITION_DEFAULTS = {
  showPhrase: true,
  showSuggsAtBos: true,
}

export const namedConditions = {
  trump: {
    sugFlags: {
      useSufarr: false,
      temperature: 0,
      use_bos_suggs: false,
      domain: 'tweeterinchief'
    },
    showPhrase: true,
  },
  sotu: {
    sugFlags: {
      useSufarr: false,
      temperature: 0,
      use_bos_suggs: false,
      domain: 'sotu'
    },
    showPhrase: true,
  },
  airbnb: {
    sugFlags: {
      useSufarr: false,
      temperature: 0,
      use_bos_suggs: false,
      domain: 'airbnb_train'
    },
    showPhrase: true,
  },
  nosugg: {
    sugFlags: {
      useSufarr: false,
      temperature: 0,
      continuation_length: 0,
    },
    showPhrase: false,
    hideFullwordPredictions: true,
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
  samplePhrase: {
    sugFlags: {
      temperature: 1.,
      useSufarr: false,
      rare_word_bonus: 0.,
      continuation_length: 17
    },
    showPhrase: true
  },
  topicdiverse: {
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
  topiccontinue: {
    sugFlags: {
      useSufarr: false,
      rare_word_bonus: 0.,
      null_logprob_weight: 0.,
      use_bos_suggs: 'continue',
      continuation_length: 17,
    },
    showPhrase: true
  },
  handcluster: {
    sugFlags: {
      use_bos_suggs: 'manual',
      continuation_length: 17
    },
    showPhrase: true,
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
  },
  sentdiverse: {
    sugFlags: {
      useSufarr: false,
      continuation_length: 17,
      use_bos_suggs: false,
    },
    showPhrase: true,
    showSuggsAtBos: true,
    sentiment: 'diverse',
  },
  sentpos: {
    sugFlags: {
      useSufarr: false,
      continuation_length: 17,
      use_bos_suggs: false,
    },
    showPhrase: true,
    showSuggsAtBos: true,
    sentiment: 5,
    // useAttentionCheck: .1,
  },
  sentneg: {
    sugFlags: {
      useSufarr: false,
      continuation_length: 17,
      use_bos_suggs: false,
    },
    showPhrase: true,
    showSuggsAtBos: true,
    sentiment: 1,
    // useAttentionCheck: .1,
  },
  sentmatch: {
    sugFlags: {
      useSufarr: false,
      continuation_length: 17,
      use_bos_suggs: false,
    },
    showPhrase: true,
    showSuggsAtBos: true,
    sentiment: 'match',
  },
  yelppredict: {
    sugFlags: {
      split: true,
      num_sims: 0,
      num_alternatives: 0,
    },
    showSynonyms: false,
    showReplacement: false,
    // useAttentionCheck: .1,
    hideFullwordPredictions: true,
  },
  yelpalternatives: {
    sugFlags: {
      split: true,
      num_sims: 10,
      num_alternatives: 3,
    },
    showSynonyms: true,
    showReplacement: true,
    // useAttentionCheck: .1,
    hideFullwordPredictions: true,
  },
  airbnbAlternatives: {
    sugFlags: {
      split: true,
      num_sims: 10,
      num_alternatives: 3,
      domain: 'airbnb_train'
    },
    showSynonyms: true,
    showReplacement: true,
    hideFullwordPredictions: true,
  },
  airbnbPlain: {
    sugFlags: {
      split: true,
      num_sims: 10,
      num_alternatives: 5,
      domain: 'airbnb_train'
    },
    showSynonyms: false,
    showReplacement: false,
    hideFullwordPredictions: true,
  },
  pressandhold: {
    sugFlags: {
      alternatives: true,
    },
    alternatives: true
  },

  yelprare: {
    sugFlags: {
      continuation_length: 17,
      domain: 'yelp_lowfreq'
    },
  },

  yelpcommon: {
    sugFlags: {
      continuation_length: 17,
      domain: 'yelp_hifreq'
    },
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
  topicdiversity: {
    baseConditions: ['topicdiverse', 'topiccontinue'],
    prewrite: false,
    instructions: 'tabooTopic'
  },
  infoSource: {
    baseConditions: ['withPrewrite', 'phrase'],
    prewrite: false,
    instructions: 'review'
  },
  sentiment: {
    baseConditions: ['sentdiverse', 'nosugg'],
    prewrite: false,
    instructions: 'sentiment'
  },
  sent3: {
    baseConditions: ['sentdiverse', 'sentmatch', 'nosugg'],
    instructions: 'sentiment'
  },
  polarized: {
    baseConditions: ['sentpos', 'sentneg', 'word'],
    instructions: 'review',
  },
  sent32: {
    baseConditions: ['sentdiverse', 'sentmatch', 'word'],
    instructions: 'review',
  },
  sent4: {
    baseConditions: ['sentpos', 'sentneg'],
    instructions: 'yelp',
    timeEstimate: '45 to 75 minutes',
  },
  synonyms: {
    baseConditions: ['yelppredict', 'yelpalternatives'],
    instructions: 'yelp',
    showAlternativesPractice: true,
  }
};



function getScreens(masterConfigName: string, conditions: string[]) {
  let masterConfig = MASTER_CONFIGS[masterConfigName];
  let {showAlternativesPractice} = masterConfig;
  let tutorialCondition = showAlternativesPractice ? 'airbnbPlain' : 'airbnb';
  let result = [
    {controllerScreen: 'Welcome', screen: 'Welcome'},
    {screen: 'SelectRestaurants'},
    {preEvent: {type: 'setupExperiment', block: 0, condition: tutorialCondition, name: 'practice'}, screen: 'ExperimentScreen', controllerScreen: 'PracticeComputer'},
  ];
  if (showAlternativesPractice) {
    result.push({preEvent: {type: 'setupExperiment', block: 0, condition: 'airbnbAlternatives', name: 'practice-2'}, screen: 'ExperimentScreen', controllerScreen: 'PracticeAlternativesInstructions'});
  } else {
    result.push({preEvent: {type: 'setupExperiment', block: 0, condition: tutorialCondition, name: 'practice-2'}, screen: 'ExperimentScreen', controllerScreen: 'TutorialInstructions'});
  }
  if (masterConfigName === 'infoSource') {
    conditions.forEach((conditionName, block) => {
      result = result.concat([
        {preEvent: {type: 'setupExperiment', block, condition: conditionName, name: `final-${block}`}, controllerScreen: 'ListWords'},
        {screen: 'ExperimentScreen', controllerScreen: 'RevisionComputer', timer: finalTimer},
        {controllerScreen: 'PostTaskSurvey'},
      ]);
    });

  } else {
    conditions.forEach((condition, idx) => {
      result = result.concat(experimentBlock(idx, condition, masterConfig.prewrite));
    });
  }
  result = result.concat([
    {controllerScreen: 'ShowReviews', screen: 'PostExpSurvey'},
    {screen: 'IntroSurvey'},
    {screen: 'Done', controllerScreen: 'Done'},
  ]);
  return result;
}


function specialSent4PlaceSort(participantId, places) {
  let pos = places.slice(0, 2), neg = places.slice(2);
  neg = seededShuffle(`${participantId}-places-neg`, neg);
  pos = seededShuffle(`${participantId}-places-pos`, pos);
  let order = seededShuffle(`${participantId}-places-order`, [pos, neg]);
  return order[0].concat(order[1]);
}

// function testSpecialSent4PlaceSort() {
//   JSON.stringify(specialSent4PlaceSort('a01', [{stars: 5}, {stars: 4}, {stars: 1}, {stars: 3}]));
// }

export class MasterStateStore {
  masterConfig: Object;
  masterConfigName: string;
  prewrite: boolean;
  clientId: string;
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
    this.conditions = seededShuffle(`${this.clientId}-conditions`, this.masterConfig.baseConditions);
    if (configName === 'sent4') {
      // This condition repeats the base conditions.
      this.conditions = this.conditions.concat(this.conditions);
    }
    this.screenNum = 0;
  }

  constructor(clientId:string) {
    this.clientId = clientId;

    let isDemo = (clientId || '').slice(0, 4) === 'demo';
    this.isDemo = isDemo;
    this.demoConditionName = clientId.slice(4);
    let sentiment = null;
    if (this.demoConditionName.slice(0,9) === 'sentmatch') {
      sentiment = +this.demoConditionName[9];
      this.demoConditionName = 'sentmatch';
    }

    this.times = {prewriteTimer, finalTimer};

    M.extendObservable(this, {
      masterConfig: null,
      participantCode: null,
      get isHDSL() {
        return this.participantCode !== null;
      },
      get isMTurk() {
        return !this.isHDSL;
      },
      get timeEstimate() { return this.masterConfig.timeEstimate; },
      get prewrite() { return this.masterConfig.prewrite; },
      prewriteText: '',
      curPrewriteLine: 0,
      get prewriteLines() {
        let text = this.prewriteText.trim();
        if (text === '') return [];
        return text.split('\n');
      },
      sentiment,
      lastEventTimestamp: null,
      replaying: true,
      screenNum: null,
      block: null,
      conditions: null,
      conditionName: null,
      get isPractice() {
        return (this.curExperiment || '').slice(0, 5) === 'pract';
      },
      experiments: M.observable.shallowMap({}),
      curExperiment: null,
      get experimentState() {
        if (this.curExperiment) {
          return this.experiments.get(this.curExperiment);
        }
      },
      get isPrewrite() {
        return this.curExperiment.slice(0, 3) === 'pre';
      },
      controlledInputs: M.observable.shallowMap({}),
      timerStartedAt: null,
      timerDur: null,
      tutorialTasks: null,
      screenTimes: [],
      passedQuiz: false,
      lastFailedAttnCheck: 0,
      get showAttnCheckFailedMsg() {
        return this.lastEventTimestamp && this.lastEventTimestamp - this.lastFailedAttnCheck < 3000;
      },
      phoneSize: {width: 360, height: 500},
      pingTime: null,
      get screens() {
        if (isDemo) {
          return [{
            preEvent: {type: 'setupExperiment', block: 0, condition: this.demoConditionName, name: 'demo'},
            screen: 'ExperimentScreen', controllerScreen: 'ExperimentScreen'
          }];
        }
        return getScreens(this.masterConfigName, this.conditions);
      },
      get curScreen() {
        if (this.screenNum) {
          return this.screens[this.screenNum];
        } else {
          return {};
        }
      },
      get places() {
        let {controlledInputs} = this;
        let res = this.conditions.map((condition, idx) => ({
          idx,
          name: controlledInputs.get(`restaurant${idx+1}`),
          visit: controlledInputs.get(`visit${idx+1}`) + " day(s) ago",
          stars: controlledInputs.get(`star${idx+1}`),
          knowWhatToWrite: controlledInputs.get(`knowWhat${idx+1}`)
        }));
        if (this.masterConfigName === 'sent4') {
          // Sort places special: pick two good, then two bad, or vice versa.
          return specialSent4PlaceSort(this.clientId, res);
        } else {
          return seededShuffle(`${this.clientId}-places`, res);
        }
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
        return {...CONDITION_DEFAULTS, ...namedConditions[this.conditionName]};
      },
    });
  }

  initScreen() {
    // Execute start-of-screen actions.
    let sideEffects = [];
    let screen = this.screens[this.screenNum];
    if (screen.preEvent) {
      let {preEvent} = screen;
      switch (preEvent.type) {
      case 'setupExperiment':
        this.block = preEvent.block;
        this.conditionName = preEvent.condition;
        this.curExperiment = preEvent.name;

        let sentiment = this.condition.sentiment;
        if (sentiment === 'match') {
          sentiment = this.sentiment || this.curPlace.stars;
        }
        let sugFlags = {
          domain: 'yelp_train-balanced',
          ...this.condition.sugFlags,
          sentiment
        };

        let experimentObj = new ExperimentStateStore(this.condition, sugFlags);
        this.experiments.set(preEvent.name, experimentObj);
        sideEffects.push(experimentObj.init());
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
    return sideEffects;
  }

  handleEvent = M.action((event) => {
    let sideEffects = [];
    this.lastEventTimestamp = event.jsTimestamp;
    if (this.experimentState) {
      let res = this.experimentState.handleEvent(event);
      if (res) sideEffects = sideEffects.concat(res);
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
      if (this.isDemo && !this.experimentState) {
        this.setMasterConfig('demo');
        if (this.demoConditionName === 'withPrewrite') {
          this.prewriteText = "tacos\nbest food\ncheap\nextra queso\ncarne asada\nweekly\ngood place for pick up not eat in\nwalls echo\nalways same order\nnew try horchata\n"
        }
        this.pingTime = 0;
      }
    break;
    case 'pingResults':
      if (event.kind === 'p') {
        this.pingTime = event.ping.mean;
      }
      break;
    case 'failedAttnCheckForce':
      if (!this.replaying) {
        this.lastFailedAttnCheck = event.jsTimestamp;
      }
      break;
    case 'passedAttnCheck':
      this.lastFailedAttnCheck = 0;
      break;

    default:
    }

    if (this.screenNum !== screenAtStart) {
      sideEffects = sideEffects.concat(this.initScreen());
    }
    return sideEffects;
  });
}
