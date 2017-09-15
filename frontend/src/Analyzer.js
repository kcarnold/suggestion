import * as M from "mobx";
import _ from "lodash";

const INCOMPLETE_BUT_OK = 'hfj33r'.split(/s/);

export function processLogGivenStateStore(StateStoreClass, log) {
  let { participant_id } = log[0];
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
        finalText: "",
        actions: [],
        annotatedFinalText: [],
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
  let lastDisplayedSuggs = null;

  log.forEach((entry, logIdx) => {
    // We need to track context sequence numbers instead of curText because
    // autospacing after punctuation seems to increment contextSequenceNum
    // without changing curText.
    let lastContextSeqNum = (state.experimentState || {}).contextSequenceNum;
    let lastText = (state.experimentState || {}).curText;
    let suggestionContext = (state.experimentState || {}).suggestionContext;

    let isValidSugUpdate =
      entry.type === "receivedSuggestions" &&
      entry.msg.request_id === (state.experimentState || {}).contextSequenceNum;

    // Track requests
    if (entry.kind === "meta" && entry.type === "requestSuggestions") {
      let msg = _.clone(entry.request);
      let requestCurText =
        msg.sofar + msg.cur_word.map(ent => ent.letter).join("");
      requestsByTimestamp[msg.timestamp] = { request: msg, response: null };
      if (tmpSugRequests[msg.request_id]) {
        console.assert(
          tmpSugRequests[msg.request_id] === requestCurText,
          `Mismatch request curText for ${participant_id}-${msg.timestamp}}, "${tmpSugRequests[
            msg.request_id
          ]}" VS "${requestCurText}"`,
        );
        // console.log("Ignoring duplicate request", msg.timestamp);
        requestsByTimestamp[msg.timestamp].dupe = true;
        return;
      } else {
        tmpSugRequests[msg.request_id] = requestCurText;
      }
    } else if (entry.type === "receivedSuggestions") {
      let msg = { ...entry.msg, responseTimestamp: entry.jsTimestamp };
      requestsByTimestamp[msg.timestamp].response = msg;
    }

    if (entry.kind !== "meta") {
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

    if (entry.jsTimestamp) {
      if (pageData.firstEventTimestamp === null) {
        pageData.firstEventTimestamp = entry.jsTimestamp;
      }
      pageData.lastEventTimestamp = entry.jsTimestamp;
    }

    let annotatedAction = {};
    if (!lastText) {
      lastText = '';
    }

    if (
      [
        "connected",
        "init",
        "requestSuggestions",
        "receivedSuggestions",
        "next",
      ].indexOf(entry.type) === -1
    ) {
      let {curWord} = suggestionContext;
      let annoType = entry.type;
      if (entry.type === 'tapSuggestion') {
        let trimtext = lastText.trim();
        if (trimtext.length === 0 || trimtext.match(/[.?!]$/)) {
          annoType = 'tapSugg_bos';
        } else if (curWord.length === 0) {
          annoType = 'tapSugg_full';
        } else {
          annoType = 'tapSugg_part';
        }
      }
      annotatedAction = {
        ...entry,
        annoType,
        curText: lastText,
        timestamp: entry.jsTimestamp,
        visibleSuggestions: lastDisplayedSuggs,
      };
      if (entry.type === 'tapSuggestion') {
        annotatedAction.sugInserted = lastDisplayedSuggs[entry.which][entry.slot].words[0].slice(curWord.length);
      }
      pageData.actions.push(annotatedAction);
    }

    let {curText} = expState;
    let {annotatedFinalText} = pageData;
    if (lastText !== curText) {
      // Update the annotation.
      let commonPrefixLen = Math.max(0, lastText.length - 10);
      while (lastText.slice(0, commonPrefixLen) !== curText.slice(0, commonPrefixLen)) {
        commonPrefixLen--;
      }
      while (lastText.slice(0, commonPrefixLen + 1) === curText.slice(0, commonPrefixLen + 1)) {
        commonPrefixLen++;
      }
      annotatedFinalText.splice(commonPrefixLen, lastText.length - commonPrefixLen);
      Array.prototype.forEach.call(curText.slice(commonPrefixLen), char => {
        annotatedFinalText.push({char, action: annotatedAction});
      });
    }

    let visibleSuggestions = M.toJS(expState.visibleSuggestions);
    if (expState.contextSequenceNum !== lastContextSeqNum) {
      if (pageData.displayedSuggs[lastContextSeqNum]) {
        pageData.displayedSuggs[lastContextSeqNum].action = entry;
      }
      lastContextSeqNum = expState.contextSequenceNum;
    } else if (entry.type === "receivedSuggestions" && isValidSugUpdate) {
      let { request, response } = requestsByTimestamp[entry.msg.timestamp];
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

    if (
      pageData.displayedSuggs[expState.contextSequenceNum] &&
      !_.isEqual(visibleSuggestions, lastDisplayedSuggs)
    ) {
      pageData.displayedSuggs[
        expState.contextSequenceNum
      ].recs = visibleSuggestions;
      lastDisplayedSuggs = visibleSuggestions;
    }
  });

  // Close out all the experiment pages.
  pageSeq.forEach(pageName => {
    let pageData = byExpPage[pageName];
    let expState = state.experiments.get(pageName);
    pageData.finalText = expState.curText;
    pageData.displayedSuggs[pageData.displayedSuggs.length - 1].action = {
      type: "next",
    };
    pageData.secsOnPage =
      (pageData.lastEventTimestamp - pageData.firstEventTimestamp) / 1000;

    let {annotatedFinalText} = pageData;
    delete pageData['annotatedFinalText'];
    let lastAction = null;
    let chunks = [];
    annotatedFinalText.forEach(({char, action}) => {
      if (action !== lastAction) {
        chunks.push({chars: char, action, timestamp: action.jsTimestamp, actionClass: action.annoType});
        lastAction = action;
      } else {
        chunks[chunks.length - 1].chars += char;
      }
    });
    console.assert(chunks.map(x => x.chars).join('') === pageData.finalText);
    pageData.chunks = chunks;

    // Group chunks into words.
    let words = [{chunks: []}];
    chunks.forEach(chunk => {
      words[words.length - 1].chunks.push(chunk);

      let {chars} = chunk;
      let endsWord = chars.match(/[-\s.!?,]/);
      if (endsWord) {
        words.push({chunks: []});
      }
    });
    words = words.filter(x => x.chunks.length > 0);
    pageData.words = words;
  });

  // One log didn't get to the last
  if (INCOMPLETE_BUT_OK.indexOf(participant_id) === -1) {
    console.assert(
      state.curScreen.screen === "Done" ||
        state.curScreen.screen === "IntroSurvey",
      "Incomplete log file %s (on screen %s)",
      participant_id,
      state.curScreen.screen || state.curScreen.controllerScreen
    );
  }

  let screenTimes = state.screenTimes.map(screen => {
    let screenDesc = state.screens[screen.num];
    return {
      ...screen,
      name: screenDesc.screen || screenDesc.controllerScreen,
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
  let { rev } = log[0];
  return (await import(`../../old-code/${rev}/frontend/src/MasterStateStore`))
    .MasterStateStore;
}

export async function analyzeLog(log) {
  let stateStoreClass = await getStateStoreClass(log);
  return processLogGivenStateStore(stateStoreClass, log);
}
