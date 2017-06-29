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

const CONDITION_DEFAULTS = {
  showPhrase: true,
  showSuggsAtBos: true,
}

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
  sotu: {
    sugFlags: {
      useSufarr: false,
      temperature: 0,
      use_bos_suggs: false,
      domain: 'sotu'
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
    hideSuggUnlessPartialWord: true,
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
    useAttentionCheck: true,
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
    useAttentionCheck: true,
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
  }
};



function getScreens(masterConfigName: string, conditions: string[]) {
  let masterConfig = MASTER_CONFIGS[masterConfigName];
  let result = [
    {controllerScreen: 'Welcome', screen: 'ProbablyWrongCode'},
    {screen: 'SetupPairingPhone', controllerScreen: 'SetupPairingComputer'},
    {preEvent: {type: 'setupExperiment', block: 0, condition: 'sotu', name: 'practice'}, screen: 'PracticePhone', controllerScreen: 'PracticeComputer'},
    {controllerScreen: 'SelectRestaurants'},
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
    conditions.forEach((condition, idx) => {
      result = result.concat(experimentBlock(idx, condition, masterConfig.prewrite));
    });
  }
  result = result.concat([
    {screen: 'ShowReviews', controllerScreen: 'PostExpSurvey'},
    {screen: 'Done', controllerScreen: 'Done'},
  ]);
  return result;
}


function shuffle(rng, array) {
  // Fisher-Yates shuffle, with a provided RNG function.
  // Basically: build up a shuffled part at the end of the array
  // by swapping the last unshuffled element with a random earlier one.
  // See https://bost.ocks.org/mike/shuffle/ for a nice description

  // First, copy the array (bostock's impl forgets this).
  array = Array.prototype.slice.call(array);

  let m = array.length;
  while(m) {
    // Pick an element from the part of the list that's not yet shuffled.
    let prevElement = Math.floor(rng() * m--);

    // Swap it with the current element.
    let tmp = array[prevElement];
    array[prevElement] = array[m];
    array[m] = tmp;
  }
  return array;
}

function seededShuffle(seed, array) {
  return shuffle(seedrandom(seed), array);
}

function specialSent4PlaceSort(participantId, places) {
  places = _.sortBy(places, 'stars');
  let neg = places.slice(0, 2), pos = places.slice(2);
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
    this.initScreen();
  }

  constructor(clientId:string) {
    this.clientId = clientId;

    let isDemo = (clientId || '').slice(0, 4) === 'demo';
    let demoConditionName = clientId.slice(4);
    let sentiment = null;
    if (demoConditionName.slice(0,9) === 'sentmatch') {
      sentiment = +demoConditionName[9];
      demoConditionName = 'sentmatch';
    }

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
      sentiment,
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
        let res = this.conditions.map((condition, idx) => ({
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
      get suggestionRequestParams() {
        let sentiment = this.condition.sentiment;
        if (sentiment === 'match') {
          sentiment = this.sentiment || this.curPlace.stars;
        }
        return {
          domain: 'yelp_train-balanced',
          ...this.condition.sugFlags,
          sentiment
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
        return {...CONDITION_DEFAULTS, ...namedConditions[this.conditionName]};
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
        this.prewriteText = "tacos\nbest food\ncheap\nextra queso\ncarne asada\nweekly\ngood place for pick up not eat in\nwalls echo\nalways same order\nnew try horchata\n"
      }
      this.pingTime = 0;
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
        this.conditionName = preEvent.condition;
        this.curExperiment = preEvent.name;
        this.experiments.set(preEvent.name, new ExperimentStateStore(this.condition));
        this.tutorialTasks = new TutorialTasks();
        if (this.masterConfig.useConstraints) {
          this.experimentState.useConstraints = this.masterConfig.useConstraints;
        }
        this.experimentState.showSuggsAtBos = this.condition.showSuggsAtBos;
        this.experimentState.hideSuggUnlessPartialWord = this.condition.hideSuggUnlessPartialWord;
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
    return sideEffects;
  });
}
