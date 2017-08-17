import {MasterStateStore} from './MasterStateStore';
import _ from 'lodash';

export function processLog(log) {
  let {participant_id} = log[0];
  let state = new MasterStateStore(participant_id);
  let byExpPage = {};
  let pageSeq = [];
  let requestsByTimestamp = {};

  log.forEach((entry) => {
    if (entry.kind === 'meta') return;
    let {participant_id} = entry;
    let lastText = (state.experimentState || {}).curText;
    let isValidSugUpdate = entry.request_id === (state.experimentState || {}).contextSequenceNum;
    state.handleEvent(entry);
    if (state.experimentState) {
      let expState = state.experimentState;
      let page = state.curExperiment;
      let pageData = byExpPage[page];
      if (!pageData) {
        pageData = {
          annotated: [],
          displayedSuggs: [],
          condition: state.conditionName,
        };
        byExpPage[page] = pageData;
        pageSeq.push(page);
      }
      if (expState.curText !== lastText) {
        pageData.annotated.push({...entry, curText: lastText});
        lastText = state.curText;
        if (pageData.displayedSuggs.length > 0) {
          let lastDisplayedSugg = pageData.displayedSuggs[pageData.displayedSuggs.length - 1];
          lastDisplayedSugg.dieEvent = event;
        }
      } else if (entry.type === 'receivedSuggestions' && isValidSugUpdate) {
        experiment.displayedSuggs.push({
          visibleSuggestions: expState.visibleSuggestions,
          displayEvent: entry,
          dieEvent: null,
        });
      }
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

export function processCombinedLog(log) {
  return _.map(
    _.groupBy(log, 'participant_id'),
    (entries, participant_id) => [participant_id, processLog(entries)]);
}
