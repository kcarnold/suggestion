import * as M from 'mobx';
import _ from 'lodash';

let HACK_TSCODES_TO_SKIP = {}
const CODES = "p964wg-1504799690416";
CODES.split(/\s/).forEach(code => {
  HACK_TSCODES_TO_SKIP[code] = true;
});

export function processLogGivenStateStore(StateStoreClass, log) {
  let {participant_id} = log[0];
  let state = new StateStoreClass(participant_id);
  let byExpPage = {};
  let pageSeq = [];
  let requestsByTimestamp = {};

  function getPageData() {
    let page = state.curExperiment;
    if (!byExpPage[page]) {
      let pageData = {
        displayedSuggs: [],
        condition: state.conditionName,
        place: state.curPlace,
        finalText: '',
        actions: [],
        firstEventTimestamp: null,
        lastEventTimestamp: null,
      };
      byExpPage[page] = pageData;
      pageSeq.push(page);
    }
    return byExpPage[page];
  }

  let lastScreenNum = null;
  let tmpSugRequests = null;
  let lastSugResponseTimestamp = null;
  let stateMismatches = [];

  log.forEach((entry, logIdx) => {

    // We need to track context sequence numbers instead of curText because
    // autospacing after punctuation seems to increment contextSequenceNum
    // without changing curText.
    let lastContextSeqNum = (state.experimentState || {}).contextSequenceNum;
    let lastText = (state.experimentState || {}).curText;
    let lastDisplayedSuggs = null;

    let isValidSugUpdate = entry.type === 'receivedSuggestions' && entry.msg.request_id === (state.experimentState || {}).contextSequenceNum;

    // Track requests
    if (entry.kind === 'meta' && entry.type === 'requestSuggestions') {
      let msg = _.clone(entry.request);
      requestsByTimestamp[msg.timestamp] = {request: msg, response: null};
      if (tmpSugRequests[msg.request_id]) {
        console.log("Ignoring duplicate request", msg.timestamp);
        requestsByTimestamp[msg.timestamp].dupe = true;
        return;
      } else {
        tmpSugRequests[msg.request_id] = 'request';
      }
    } else if (entry.type === 'receivedSuggestions') {
      let msg = {...entry.msg, responseTimestamp: entry.jsTimestamp};
      let tscode = `${participant_id}-${msg.timestamp}`;
      if (HACK_TSCODES_TO_SKIP[tscode]) {
        return;
      }
      lastSugResponseTimestamp = tscode;
      if (false && requestsByTimestamp[msg.timestamp].dupe) {
        console.log("Ignoring response to duplicate request", msg.timestamp);
        return;
      } else {
        requestsByTimestamp[msg.timestamp].response = msg;
        tmpSugRequests[msg.request_id] = 'response';
      }
    }


    if (entry.kind !== 'meta') {
      // if (entry.type !== 'receivedSuggestions' || isValidSugUpdate)
      state.handleEvent(entry);
    }

    if (state.screenNum !== lastScreenNum) {
      tmpSugRequests = {};
      lastScreenNum = state.screenNum;
    }

    let expState = state.experimentState;
    if (!expState) {
      return;
    }

    let pageData = getPageData();

    if (pageData.firstEventTimestamp === null) {
      pageData.firstEventTimestamp = entry.jsTimestamp;
    }
    pageData.lastEventTimestamp = entry.jsTimestamp;

    // Assert state consistency
    if (entry.kind === 'meta' && entry.type === 'requestSuggestions' && entry.request.request_id === expState.contextSequenceNum) {
      let requestCurText = entry.request.sofar + entry.request.cur_word.map((ent => ent.letter)).join('');
      if (requestCurText !== expState.curText) {
        stateMismatches.push(lastSugResponseTimestamp);
        console.log(participant_id, logIdx, "Correcting curText:", expState.curText, 'TO', requestCurText)
        expState.curText = requestCurText;
        // throw new Error(`State mismatch! ${entry.request.sofar} vs ${expState.suggestionContext.prefix} - last sug response ${lastSugResponseTimestamp}`)
      }
    }

    if (['connected', 'init', 'requestSuggestions', 'receivedSuggestions'].indexOf(entry.type) === -1) {
      pageData.actions.push({...entry, curText: lastText, timestamp: entry.jsTimestamp});
    }

    let visibleSuggestions = M.toJS(expState.visibleSuggestions);
    if (expState.contextSequenceNum !== lastContextSeqNum) {
      if (pageData.displayedSuggs[lastContextSeqNum]) {
        pageData.displayedSuggs[lastContextSeqNum].action = entry;
      }
      lastContextSeqNum = expState.contextSequenceNum;
    } else if (entry.type === 'receivedSuggestions' && isValidSugUpdate) {
      let {request, response}  = requestsByTimestamp[entry.msg.timestamp];
      pageData.displayedSuggs[expState.contextSequenceNum] = {
        request_id: request.request_id,
        sofar: request.sofar,
        cur_word: request.cur_word,
        flags: request.flags,
        timestamp: request.timestamp,
        context: expState.curText,
        recs: visibleSuggestions,
        latency: response.responseTimestamp - request.timestamp,
        action: null,
      };
    }

    if (pageData.displayedSuggs[expState.contextSequenceNum] && !_.isEqual(visibleSuggestions, lastDisplayedSuggs)) {
      pageData.displayedSuggs[expState.contextSequenceNum].recs = visibleSuggestions;
      lastDisplayedSuggs = visibleSuggestions;
    }
  });

  // Close out all the experiment pages.
  pageSeq.forEach(pageName => {
    let pageData = byExpPage[pageName];
    let expState = state.experiments.get(pageName);
    pageData.finalText = expState.curText;
    pageData.displayedSuggs[pageData.displayedSuggs.length - 1].action = {type: 'next'};
    pageData.secsOnPage = (pageData.lastEventTimestamp - pageData.firstEventTimestamp) / 1000;
  });

  if (stateMismatches.length) {
    console.error(stateMismatches.join(' '));
    throw new Error(`State mismatches: ${stateMismatches.join(' ')}`);
  }

  console.assert(state.curScreen.screen === 'Done', "Incomplete log file %s", participant_id);

  let screenTimes = state.screenTimes.map(screen => {
    let screenDesc = state.screens[screen.num];
    return {
      ...screen, name: screenDesc.screen || screenDesc.controllerScreen
    };
  });

  return {
    participant_id,
    config: state.masterConfigName,
    byExpPage,
    pageSeq,
    screenTimes,
    conditions: state.conditions,
  };
}

async function getStateStoreClass(log) {
  let {rev} = log[0];
  return (await import(`../../old-code/${rev}/frontend/src/MasterStateStore`)).MasterStateStore;
}

export async function analyzeLog(log) {
  let stateStoreClass = await getStateStoreClass(log);
  return processLogGivenStateStore(stateStoreClass, log);
}
