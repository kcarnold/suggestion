import { ExperimentStateStore } from "./ExperimentState";
import * as M from "mobx";

const recs0 = {
  predictions: [{ words: ["of"] }, { words: ["and"] }, { words: ["the"] }],
  replacement_range: [4, 10],
  synonyms: [
    { words: ["front"] },
    { words: ["interior"] },
    { words: ["exterior"] },
  ],
};

const recs1 = {
  predictions: [
    {
      words: ["one", "of", "my", "favorite", "places"],
      meta: {
        llk: -7.9542345805681505,
        sentiment_summary: 0.529547463330704,
        bos: true,
      },
    },
    {
      words: ["this", "is", "my", "favorite", "place"],
      meta: {
        llk: -8.124277723984124,
        sentiment_summary: 0.5043874292873374,
        bos: true,
      },
    },
    {
      words: ["i", "love", "this", "place", ","],
      meta: {
        llk: -8.021104056520906,
        sentiment_summary: 0.44152837517322413,
        bos: true,
      },
    },
  ],
};

function tapKeys(state, keys) {
  Array.prototype.forEach.call(keys, key =>
    state.handleEvent({ type: "tapKey", key }),
  );
}

it("inserts automatic spaces after suggestions", () => {
  var state = new ExperimentStateStore({});
  const curText = "the inside ";
  tapKeys(state, curText);
  expect(state.curText).toEqual(curText);

  state.handleEvent({
    type: "receivedSuggestions",
    msg: { request_id: state.contextSequenceNum, ...recs0 },
  });
  expect(state.visibleSuggestions.replacement_range).toEqual(
    recs0.replacement_range,
  );
  expect(state.visibleSuggestions.synonyms).toEqual(recs0.synonyms);

  state.handleEvent({ type: "tapSuggestion", which: "synonyms", slot: 1 });
  expect(state.curText).toEqual("the interior ");

  state.handleEvent({ type: "tapKey", key: "." });
  expect(state.curText).toEqual("the interior. ");
});

it("promises a phrase completion even without a server roundtrip", () => {
  let state = new ExperimentStateStore({});
  let words = ["this", "is", "my", "favorite", "place"];
  state.handleEvent({
    type: "receivedSuggestions",
    msg: { request_id: state.contextSequenceNum, ...recs1 },
  });
  expect(M.toJS(state.visibleSuggestions.predictions[1].words)).toEqual(words);

  state.handleEvent({ type: "tapSuggestion", which: "predictions", slot: 1 });
  expect(state.curText).toEqual("this ");

  expect(state.activeSuggestion).not.toEqual(null);

  expect(state.suggestionContext.promise.slot).toEqual(1);
  expect(M.toJS(state.suggestionContext.promise.words)).toEqual(words.slice(1));

  expect(M.toJS(state.visibleSuggestions.predictions[1].words)).toEqual(
    words.slice(1),
  );
});

it("promises the same word completion as long as the user is typing a prefix", () => {
  let state = new ExperimentStateStore({});
  let slot = 1, otherSlot = 0;
  let words = recs1.predictions[slot].words; //["this", "is", "my", "favorite", "place"];

  // Set up the received recommendations
  state.handleEvent({
    type: "receivedSuggestions",
    msg: { request_id: state.contextSequenceNum, ...recs1 },
  });
  expect(M.toJS(state.visibleSuggestions.predictions[slot].words)).toEqual(
    words,
  );

  // Start typing that word.
  let word = words[0];
  Array.prototype.forEach.call(word, (key, charIdx) => {
    state.handleEvent({ type: "tapKey", key });
    // The corresponding slot should still display this word.
    let shownInSlot = state.visibleSuggestions.predictions[slot];
    expect(shownInSlot.words[0]).toEqual(word);
    expect(state.visibleSuggestions.predictions[otherSlot].words).toEqual([]);
    expect(shownInSlot.highlightChars).toEqual(charIdx + 1);
    expect(state.suggestionContext.promise.slot).toEqual(1);
    expect(M.toJS(state.suggestionContext.promise.words)).toEqual(words);
  });

  // Typing space clears it.
  state.handleEvent({ type: "tapKey", key: ' '});
  expect(state.visibleSuggestions.predictions[slot].words).toEqual([]);
});

it("doesn't promise the same word completion if the user isn't typing a prefix", () => {
  let state = new ExperimentStateStore({});
  let slot = 1;
  let words = recs1.predictions[slot].words; //["this", "is", "my", "favorite", "place"];

  // Set up the received recommendations
  state.handleEvent({
    type: "receivedSuggestions",
    msg: { request_id: state.contextSequenceNum, ...recs1 },
  });
  expect(M.toJS(state.visibleSuggestions.predictions[slot].words)).toEqual(
    words,
  );

  // Start typing that word.
  let word = words[0];

  // None of the words start with "a"
  recs1.predictions.forEach(pred => {
    expect(pred.words[0][0]).not.toEqual('a');
  });

  // Tap 'a'.
  state.handleEvent({ type: "tapKey", key: 'a' });
  expect(state.visibleSuggestions.predictions[slot].words).toEqual([]);
});
