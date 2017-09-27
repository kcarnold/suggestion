import * as M from 'mobx';
import _ from 'lodash';
import countWords from './CountWords';
import seedrandom from 'seedrandom';

/**** Main experiment-screen state store!

This represents all the state relevant to a trial.

A Suggestion is an important object. It contains:
{
  // The context in which it was made
  contextSequenceNum: int
  prefix: string : characters before this suggestion
  tapLocations: array of {x: int, y: int}: cur-word taps
  words: array of string: the words of the suggestion
  meta: metadata about this suggestion (e.g., if it's a beginning-of-sentence suggestion)
  isValid: for a visible suggestion, whether it's valid in the current suggestion context.
}

We keep a list of the current suggestions. If a suggestion's sequence number doesn't match, we can still show it (so as to not be jarring), but dim it, and don't allow acceptance.

After tapping a suggestion, it becomes *active*, which means that if we keep tapping on that same spot, subsequent words will get inserted. To do this, we keep a notion of an `activeSuggestion`, which is separate from the suggestions returned from the server:
{
  suggestion: the Suggestion object
  slot: the slot number where it lives
  wordIdx: the index of the next word to insert.
}

We're not caching because (1) caching is a Hard Problem and (2) key taps will be at different places anyway, so caching won't really help.
It may be nice to prefetch what happens when a suggestion gets pressed, because there are only 3 states there and we know them exactly.
But TODO.

How this gets updated:
An autorun thread watches contextSequenceNum and requests suggestions when the context changes.

visibleSuggestions is a pure computation based on the last suggestions received from the server, the current context sequence number, and the active suggestion. It puts them all together, keeping track of what's valid.

*/

/*
Attention checks:
 - type: text, predictions, synonyms
 - the first time is forced: if it's not passed, then the participant has to tap it before moving on.
 - after that, failing an attention check has no effect.
*/

function randChoice(rng, choices) {
  let unif = 1;
  while (unif === 1)
    unif = rng();
  return choices[Math.floor(unif * choices.length)];
}

export class ExperimentStateStore {
  constructor(condition, sugFlags) {
    this.__version__ = 1;
    this.condition = condition;
    this.sugFlags = sugFlags;
    this.outstandingRequests = [];
    M.extendObservable(this, {
      curText: '',
      attentionCheck: null,
      attentionCheckStats: {
        text: {total: 0, passed: 0, force: false},
        predictions: {total: 0, passed: 0, force: false},
        phrases: {total: 0, passed: 0, force: false},
        synonyms: {total: 0, passed: 0, force: false},
      },
      tapLocations: [],
      contextSequenceNum: 0,
      lastSuggestionsFromServer: {},
      activeSuggestion: null,
      lastSpaceWasAuto: false,
      deleting: null,
      get wordCount() {
        return countWords(this.curText);
      },
      get visibleSuggestions() {
        let fromServer = this.lastSuggestionsFromServer;
        let serverIsValid = (this.condition.showPredictions === false) || fromServer.request_id === this.contextSequenceNum;
        if (!serverIsValid) {
          // Fill in the promised suggestion.
          let blankRec = {words: []};
          let predictions = _.range(3).map(() => blankRec);
          if (this.activeSuggestion) {
            predictions[this.activeSuggestion.slot] = M.toJS(this.activeSuggestion);
          }
          return {predictions};
        }

        // Make a copy, so we can modify.
        fromServer = M.toJS(fromServer);
        let result = {};
        if (fromServer.replacement_range)
          result.replacement_range = fromServer.replacement_range;
        ['predictions', 'synonyms'].forEach(type => {
          if (type === 'predictions' && this.condition.hideFullwordPredictions && !this.hasPartialWord) {
            result[type] = [];
          } else {
            result[type] = fromServer[type] || [];
          }
        });

        if (this.activeSuggestion && this.activeSuggestion.highlightChars) {
          // Highlight even what we receive from the server.
          // FIXME: should this happen on server? Format conversion is complicated...
          result.predictions[this.activeSuggestion.slot].highlightChars = this.activeSuggestion.highlightChars;
        }

        let {attentionCheck} = this;
        if (attentionCheck !== null && serverIsValid) {
          let {type: attentionCheckType} = attentionCheck;
          if (attentionCheckType === 'predictions' || attentionCheckType === 'synonyms') {
            let rec = result[attentionCheck.type][attentionCheck.slot];
            if (rec) {
              // [attentionCheck.slot].words.length > attentionCheck.word + 1) {
              // FIXME: this could be mutating a data structure that we don't own.
              rec.words[0] = 'æ' + rec.words[0];
              result.attentionCheckType = attentionCheck.type;
            }
          }
        }
        return result;
      },

      get lastSpaceIdx() {
        let sofar = this.curText;
        return sofar.search(/\s\S*$/);
      },

      get suggestionContext() {
        let sofar = this.curText, cursorPos = sofar.length;
        let lastSpaceIdx = this.lastSpaceIdx;
        let curWord = [];
        for (let i=lastSpaceIdx + 1; i<cursorPos; i++) {
          let chr = {letter: sofar[i]};
          if (this.tapLocations[i] !== null) {
            chr.tap = this.tapLocations[i];
          }
          curWord.push(chr);
        }
        let result = {
          prefix: sofar.slice(0, lastSpaceIdx + 1),
          curWord,
        };
        if (this.activeSuggestion) {
          result.promise = {
            slot: this.activeSuggestion.slot,
            words: this.activeSuggestion.words
          };
        }
        return result;
      },

      get hasPartialWord() {
        return this.suggestionContext.curWord.length > 0;
      },

      get showSynonyms() {
        if (this.condition.showSynonyms) {
          if (this.condition.showSynonymsXorPredictions) {
            return this.curText.length > 0 && this.suggestionContext.curWord.length === 0;
          }
          return true;
        }
        return false;
      },

      get showPredictions() {
        if (this.condition.dontRequestSuggestions || !this.condition.showPredictions) {
          return false;
        }
        if (this.condition.showSynonyms && this.condition.showSynonymsXorPredictions) {
          return !this.showSynonyms;
        }
        return true;
      },

      get showReplacement() {
        return this.showSynonyms;
      },

      spliceText: M.action((startIdx, deleteCount, toInsert, taps) => {
        if (!taps) {
          taps = _.map(toInsert, () => null);
        }
        this.curText = this.curText.slice(0, startIdx) + toInsert + this.curText.slice(startIdx + deleteCount);
        this.tapLocations = this.tapLocations.slice(0, startIdx).concat(taps).concat(this.tapLocations.slice(startIdx + deleteCount));
      }),
      insertText: M.action((toInsert, charsToDelete, taps) => {
        let cursorPos = this.curText.length;
        let newCursorPos = cursorPos - charsToDelete;
        this.curText = this.curText.slice(0, newCursorPos) + toInsert;
        this.tapLocations = this.tapLocations.slice(0, newCursorPos).concat(taps || _.map(toInsert, () => null));
      }),
      tapKey: M.action(event => {
        let ac = this.validateAttnCheck(event);
        if (ac.length) return ac;

        let oldCurWord = this.curText.slice(this.lastSpaceIdx + 1);

        let isNonWord = event.key.match(/\W/);
        let deleteSpace = this.lastSpaceWasAuto && isNonWord;
        let toInsert = event.key;
        let taps = [{x: event.x, y: event.y}];
        let autoSpace = isNonWord && event.key !== " " && event.key !== "'" && event.key !== '-';
        if (autoSpace) {
          toInsert += " ";
          taps.push({});
        }
        this.insertText(toInsert, deleteSpace ? 1 : 0, taps);
        this.lastSpaceWasAuto = autoSpace;
        let newActiveSuggestion = null;

        // If this key happened to be the prefix of a recommended word, continue that word.
        let curWord = this.curText.slice(this.lastSpaceIdx + 1);
        if (this.condition.showRelevanceHints && !isNonWord && curWord.slice(0, oldCurWord.length) === oldCurWord) {
          this.visibleSuggestions.predictions.forEach((pred, slot) => {
            if (pred.words.length === 0) return;
            if (newActiveSuggestion) return;
            if (pred.words[0].slice(0, curWord.length) === curWord) {
              newActiveSuggestion = {
                words: pred.words,
                slot,
                highlightChars: curWord.length
              };
            }
          });
        }
        this.activeSuggestion = newActiveSuggestion;

        return [];
      }),
      tapBackspace: M.action(() => {
        /* Ignore the attention check, don't count this for or against. */
        this.insertText('', 1);
        this.lastSpaceWasAuto = false;
        this.activeSuggestion = null;
        return [];
      }),
      handleTapSuggestion: M.action(event => {
        let {slot, which} = event;
        let ac = this.validateAttnCheck(event);
        if (ac.length) return ac;

        let tappedSuggestion = this.visibleSuggestions[which][slot];
        let wordToInsert = tappedSuggestion.words[0];
        if (!wordToInsert) return [];
        if (which === 'synonyms') {
          // Replace the _previous_ word.
          let [startIdx, endIdx] = this.visibleSuggestions['replacement_range'];
          // Actually, kill all remaining text.
          endIdx = this.curText.length;
          let autoSpace = endIdx === this.curText.length;
          this.spliceText(startIdx, endIdx - startIdx, wordToInsert);
          if (autoSpace) {
            // Add a space.
            this.spliceText(this.curText.length, 0, ' ');
          }
          if (this.curText.slice(-1) === ' ') {
            this.lastSpaceWasAuto = true;
          }
        } else {
          if (tappedSuggestion.words.length > 1) {
            this.activeSuggestion = {
              words: tappedSuggestion.words.slice(1),
              slot: slot,
            };
          } else {
            this.activeSuggestion = null;
          }

          let {curWord} = this.getSuggestionContext();
          let charsToDelete = curWord.length;
          let isNonWord = wordToInsert.match(/^\W$/);
          let deleteSpace = this.lastSpaceWasAuto && isNonWord;
          if (deleteSpace) {
            charsToDelete++;
          }
          this.insertText(wordToInsert + ' ', charsToDelete, null);
          this.lastSpaceWasAuto = true;
        }
        return [];
      }),

      handleSelectAlternative: M.action(event => {
        let ac = this.validateAttnCheck(event);
        if (ac.length) return ac;

        let wordToInsert = event.word;
        let {curWord} = this.getSuggestionContext();
        let charsToDelete = curWord.length;
        let isNonWord = wordToInsert.match(/^\W$/);
        let deleteSpace = this.lastSpaceWasAuto && isNonWord;
        if (deleteSpace) {
          charsToDelete++;
        }
        this.insertText(wordToInsert + ' ', charsToDelete, null);
        this.lastSpaceWasAuto = true;
        return [];
      }),

      handleTapText: M.action(event => {
        return this.validateAttnCheck(event);
      }),

      updateSuggestions: M.action(event => {
        let {msg} = event;
        // Only update suggestions if the data is valid.
        if (msg.request_id === this.contextSequenceNum) {
          this.lastSuggestionsFromServer = msg;
        }
        let idx = this.outstandingRequests.indexOf(msg.request_id);
        if (idx !== -1) {
          this.outstandingRequests.splice(idx, 1);
        }
        if (idx !== 0) {
          console.log('warning: outstandingRequests weird: looking for', msg.request_id, 'in', this.outstandingRequests);
        }
      }),

      handleDeleting: M.action(event => {
        let {msg} = event;
        if (msg.type === 'start') {
          this.attentionCheck = null;
          this.deleting = {
            liveChars: this.curText.length + msg.delta
          };
        } else if (msg.type === 'update') {
          console.assert(this.deleting);
          this.deleting.liveChars = Math.min(Math.max(0, this.curText.length + msg.delta), this.curText.length);
          // console.log(msg.delta, this.deleting.livech)
        } else if (msg.type === 'done') {
          console.assert(this.deleting);
          this.insertText('', this.curText.length - this.deleting.liveChars);
          this.lastSpaceWasAuto = false;
          this.activeSuggestion = null;
          this.deleting = null;
          return [];
        }
      })
    });
  }

  init() {
    if (this.condition.dontRequestSuggestions) return null;
    this.outstandingRequests.push(0);
    return this.getSuggestionRequest();
  }

  getSuggestionRequest() {
    let {prefix, curWord, promise} = this.getSuggestionContext();


    return {
      type: 'requestSuggestions',
      sofar: prefix,
      cur_word: curWord,
      flags: {...this.sugFlags, promise,},
      request_id: this.contextSequenceNum
    };
  }

  getSuggestionContext() {
    return this.suggestionContext;
  }

  validateAttnCheck(event) {
    if (this.attentionCheck === null) return [];
    let {type: attentionCheckType} = this.attentionCheck;
    let passed;
    if (attentionCheckType === 'text') {
      passed = event.type === 'tapText';
    } else if (attentionCheckType === 'predictions' || attentionCheckType === 'synonyms') {
      // only valid if there was a corresponding valid rec.
      if (!this.visibleSuggestions.attentionCheckType) return [];

      passed = (attentionCheckType === event.which && this.attentionCheck.slot === event.slot);
    }

    let stat = this.attentionCheckStats[attentionCheckType];
    if (passed) {
        this.attentionCheck = null;
        if (!stat.force) {
          stat.total++;
          stat.passed++;
        } else {
          stat.force = false;
        }
        return [{type: 'passedAttnCheck'}];
    } else {
      // The first time we're going to force it. Don't give them credit.
      if (stat.total === 0) {
        stat.force = true;
        return [{type: 'failedAttnCheckForce'}];
      } else {
        // Whatever, let 'em fail.
        console.assert(!stat.force);
        this.attentionCheck = null;
        stat.total++;
        return []; // {type: 'failedAttnCheck'} -- no, just do the action anyway.
      }
    }
  }

  handleEvent = (event) => {
    let textBeforeEvent = this.curText;
    let sideEffects = (() => {
      switch (event.type) {
      case 'tapKey':
        return this.tapKey(event);
      case 'tapBackspace':
        return this.tapBackspace();
      case 'receivedSuggestions':
        return this.updateSuggestions(event);
      case 'tapSuggestion':
        return this.handleTapSuggestion(event);
      case 'selectAlternative':
        return this.handleSelectAlternative(event);
      case 'tapText':
        return this.handleTapText(event);
      case 'updateDeleting':
        return this.handleDeleting(event);
      default:
      }
    })();
    sideEffects = sideEffects || [];

    if (this.curText !== textBeforeEvent) {
      this.contextSequenceNum++;
      // Update attn check
      let rng = seedrandom(this.curText + this.contextSequenceNum);
      if (this.condition.useAttentionCheck && rng() < this.condition.useAttentionCheck) {
        let ac = {};
        let choices = ['text'];
        if (this.showPredictions) {
          choices.push('predictions');
        }
        if (this.showSynonyms) {
          choices.push('synonyms');
        }
          ac.type = randChoice(rng, choices);
        let numSlots = ac.type === 'predictions' ? 3 : this.condition.sugFlags.num_alternatives;
        ac.slot = Math.floor(rng() * numSlots);
        this.attentionCheck = ac;
      } else {
        this.attentionCheck = null;
      }
    }

    if (!this.condition.dontRequestSuggestions && this.lastSuggestionsFromServer.request_id !== this.contextSequenceNum) {
      if (this.outstandingRequests.indexOf(this.contextSequenceNum) !== -1) {
        // console.log("Already requested", this.contextSequenceNum);
      } else if (this.outstandingRequests.length < 2) {
        // console.log(`event ${event.type} triggered request ${this.contextSequenceNum}`)
        sideEffects = sideEffects.concat([this.getSuggestionRequest()]);
        this.outstandingRequests.push(this.contextSequenceNum);
      } else {
        // console.log(`event ${event.type} would trigger request ${this.contextSequenceNum} but throttled ${this.outstandingRequests}`)
      }
    }

    return sideEffects;
  };
}
