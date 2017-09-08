import * as M from 'mobx';
import _ from 'lodash';

const HACK_TSCODES_TO_SKIP = {'p964wg-1504799690416': true};

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
      };
      byExpPage[page] = pageData;
      pageSeq.push(page);
    }
    return byExpPage[page];
  }

  let lastScreenNum = null;
  let tmpSugRequests = null;
  let lastSugResponseTimestamp = null;

  log.forEach((entry) => {

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
      if (entry.type !== 'receivedSuggestions' || isValidSugUpdate)
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

    // Assert state consistency
    if (entry.kind === 'meta' && entry.type === 'requestSuggestions') {
      if (entry.request.sofar !== expState.suggestionContext.prefix) {
        throw new Error(`State mismatch! ${entry.request.sofar} vs ${expState.suggestionContext.prefix} - last sug response ${lastSugResponseTimestamp}`)
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
  });

  console.assert(state.curScreen.screen === 'Done')

  return {
    participant_id,
    config: state.masterConfigName,
    byExpPage,
    pageSeq,
    screenTimes: state.screenTimes,
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
