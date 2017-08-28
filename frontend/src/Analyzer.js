import * as M from 'mobx';
import _ from 'lodash';

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
      };
      byExpPage[page] = pageData;
      pageSeq.push(page);
    }
    return byExpPage[page];
  }

  log.forEach((entry) => {

    // We need to track context sequence numbers instead of curText because
    // autospacing after punctuation seems to increment contextSequenceNum
    // without changing curText.
    let lastContextSeqNum = (state.experimentState || {}).contextSequenceNum;
    let lastDisplayedSuggs = null;

    let isValidSugUpdate = entry.type === 'receivedSuggestions' && entry.msg.request_id === (state.experimentState || {}).contextSequenceNum;
    if (entry.kind !== 'meta') {
      state.handleEvent(entry);
    }

    if (!state.experimentState) {
      return;
    }

    let pageData = getPageData();

    // Track requests
    if (entry.kind === 'meta' && entry.type === 'requestSuggestions') {
      let msg = _.clone(entry.request);
      requestsByTimestamp[msg.timestamp] = {request: msg, response: null};
    } else if (entry.type === 'receivedSuggestions') {
      let msg = {...entry.msg, responseTimestamp: entry.jsTimestamp};
      requestsByTimestamp[msg.timestamp].response = msg;
    }

    let expState = state.experimentState;
    let visibleSuggestions = M.toJS(expState.visibleSuggestions);
    if (expState.contextSequenceNum !== lastContextSeqNum) {
      if (pageData.displayedSuggs[lastContextSeqNum]) {
        pageData.displayedSuggs[lastContextSeqNum].action = entry;
      }
      lastContextSeqNum = expState.contextSequenceNum;
    } else if (entry.type === 'receivedSuggestions' && isValidSugUpdate) {
      let {request, response}  = requestsByTimestamp[entry.msg.timestamp];
      pageData.displayedSuggs[expState.contextSequenceNum] = {
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

  return {
    config: state.masterConfigName,
    byExpPage,
    pageSeq,
    screenTimes: state.screenTimes,
    conditions: state.conditions,
    blocks: state.conditions.map((condition, block) => {
      let expState = state.experiments.get(`final-${block}`) || {};
      return {
        condition: condition,
        prewriteText: (state.experiments.get(`pre-${block}`) || {}).curText,
        finalText: expState.curText,
        place: state.places[block],
        attentionCheckStats: expState.attentionCheckStats,
      };
    }),
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
