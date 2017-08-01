import {ExperimentStateStore} from './ExperimentState';

const recs0 = {
  "predictions": [{"word": "of"}, {"word": "and"}, {"word": "the"}], "replacement_range": [4, 10], "synonyms": [{"word": "front"}, {"word": "interior"}, {"word": "exterior"}]
};

it('inserts automatic spaces after suggestions', () => {
  var state = new ExperimentStateStore({});
  const curText = "the inside ";
  Array.prototype.forEach.call(curText, key => state.handleEvent({type: 'tapKey', key}));
  expect(state.curText).toEqual(curText);

  state.handleEvent({type: 'receivedSuggestions', msg: {request_id: state.contextSequenceNum, ...recs0}});
  expect(state.visibleSuggestions.replacement_range).toEqual(recs0.replacement_range);
  expect(state.visibleSuggestions.synonyms).toEqual(recs0.synonyms);

  state.handleEvent({type: 'tapSuggestion', which: 'synonyms', slot: 1});
  expect(state.curText).toEqual("the interior ");

  state.handleEvent({type: 'tapKey', key: '.'});
  expect(state.curText).toEqual("the interior. ");
});
