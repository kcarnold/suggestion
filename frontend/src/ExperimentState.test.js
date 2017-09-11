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

const recs2 = {
  predictions: [
    {
      words: ["."],
      meta: {
      },
    },
    {
      words: ["!"],
      meta: {
      },
    },
    {
      words: ["and", "it", "was"],
      meta: {
      },
    },
  ],
};


function tapKeys(state, keys) {
  Array.prototype.forEach.call(keys, key =>
    state.handleEvent({ type: "tapKey", key }),
  );
}

it("requests suggestions at appropriate times", () => {
  // Testing strategy:
  // - test that init should make first request
  // - test that all main user events cause a new request with a new request id
  // -- provide a response immediately for each one this time.
  // - test that suggestions don't get backlogged.
  // - test that once the backlog clears, suggestions get requested again.

});

it("requests suggestions on init", () => {
  let sugFlags = {domain: 'test'}
  let state = new ExperimentStateStore({}, sugFlags);
  expect(state.init()).toMatchObject({
    type: 'requestSuggestions',
    sofar: '',
    cur_word: [],
    request_id: 0,
    flags: expect.any(Object),
  });
});

it("requests suggestions following all main user events", () => {
  let sugFlags = {domain: 'test'}
  let state = new ExperimentStateStore({}, sugFlags);
  state.init();
  expect(state.contextSequenceNum).toEqual(0);
  state.handleEvent({
    type: "receivedSuggestions",
    msg: { request_id: 0, ...recs1 },
  });
  expect(state.contextSequenceNum).toEqual(0);

  // Try a sugg tap.
  let req1 = state.handleEvent({type: 'tapSuggestion', which: 'predictions', slot: 0});
  expect(req1).toHaveLength(1);
  expect(req1[0]).toMatchObject({
    type: 'requestSuggestions',
    sofar: "one ",
    cur_word: [],
    request_id: 1,
    flags: expect.any(Object),
  });
  expect(state.contextSequenceNum).toEqual(1);

  // The non-tapped suggestions should be invalid right now.
  [1, 2].forEach(i => {
    expect(state.visibleSuggestions.predictions[i].words).toEqual([]);
  })

  // Try a keypress.
  let req2 = state.handleEvent({type: 'tapKey', key: 'a'});
  expect(req2).toHaveLength(1);
  expect(req2[0].sofar).toEqual('one ');
  expect(req2[0].cur_word[0].letter).toEqual('a');
  expect(req2[0].request_id).toEqual(2);
  expect(state.contextSequenceNum).toEqual(2);

  // All suggestions should be invalid right now.
  [0, 1, 2].forEach(i => {
    expect(state.visibleSuggestions.predictions[i].words).toEqual([]);
  })

  // Try a backspace. Since there hasn't been a response in two requests,
  // this should not make a new request.
  let req3 = state.handleEvent({type: 'tapBackspace'});
  expect(req3).toHaveLength(0);

  // Now give a response to the first request.
  // (contents don't matter)
  let req4 = state.handleEvent({
    type: 'receivedSuggestions',
    msg: { request_id: 1, ...recs1 }
  });
  // Now we should get the request for the backspace.
  expect(req4).toHaveLength(1);
  expect(req4[0]).toMatchObject({
    sofar: 'one ',
    cur_word: [],
    request_id: 3,
    flags: expect.any(Object),
  });
  // But the visible suggestions should still be invalid.
  [0, 1, 2].forEach(i => {
    expect(state.visibleSuggestions.predictions[i].words).toEqual([]);
  });

  // Now the server starts catching up. Give the response for the key tap.
  let req4A = state.handleEvent({
    type: 'receivedSuggestions',
    msg: { request_id: 2, ...recs1 }
  });
  // No new requests.
  expect(req4A).toHaveLength(0);
  // Still invalid suggestions.
  [0, 1, 2].forEach(i => {
    expect(state.visibleSuggestions.predictions[i].words).toEqual([]);
  });


  // Now give the response for the backspace. They should show up now, and no new requests.
  let req5 = state.handleEvent({
    type: 'receivedSuggestions',
    msg: { request_id: 3, ...recs1 }
  });
  expect(req5).toHaveLength(0);
  [0, 1, 2].forEach(i => {
    expect(state.visibleSuggestions.predictions[i].words.length).toBeGreaterThan(0);
  });

  expect(state.contextSequenceNum).toEqual(3);

  // Try going far in the future, and we still only get one outstanding request.
  for (let i=0; i<10; i++) {
    let request = state.handleEvent({type: 'tapKey', key: 'x'});
    if (i < 2) {
      expect(request).toHaveLength(1);
      expect(request[0].request_id).toEqual(3 + 1 + i);
    } else {
      expect(request).toHaveLength(0);
    }
  }
  let req6 = state.handleEvent({
    type: 'receivedSuggestions',
    msg: { request_id: 4, ...recs1 }
  });
  expect(req6).toHaveLength(1);
  expect(state.contextSequenceNum).toEqual(13);
  expect(req6[0].request_id).toEqual(13);
});

xit("doesn't duplicate requests after an auto-space", () => {
  throw new Error('todo');
})


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
  let state = new ExperimentStateStore({}, {});
  state.init();
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

it("doesn't crash when running out of words", () => {
  let state = new ExperimentStateStore({}, {});
  state.init();
  let words = ["this", "is", "my", "favorite", "place"];
  state.handleEvent({
    type: "receivedSuggestions",
    msg: { request_id: state.contextSequenceNum, ...recs1 },
  });

  // Tap on that sugg way more times than there are words.
  for (let i=0; i<10; i++) {
    state.handleEvent({ type: "tapSuggestion", which: "predictions", slot: 1 });
  }

  expect(M.toJS(state.visibleSuggestions.predictions[1].words)).toEqual([]);
  expect(state.curText).toEqual(words.join(' ') + ' ');
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


it("doesn't promise word completions for punctuation", () => {
  Array.prototype.forEach.call('.!', char => {
    let state = new ExperimentStateStore({});

    tapKeys(state, "i have never had a bad experience ");

    // Set up the received recommendations
    state.handleEvent({
      type: "receivedSuggestions",
      msg: { request_id: state.contextSequenceNum, ...recs2 },
    });

    // Tap a period, which is also a shortcut, but shouldn't trigger a promise.
    expect(state.visibleSuggestions.predictions[0].words[0]).toEqual('.');
    state.handleEvent({ type: "tapKey", key: char });
    expect(state.visibleSuggestions.predictions[0].words).toEqual([]);
    expect(state.suggestionContext.promise).toBeUndefined();
  });
});
