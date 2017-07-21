import M from 'mobx';
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
export class ExperimentStateStore {
  constructor(condition) {
    this.__version__ = 1;
    this.condition = condition;
    M.extendObservable(this, {
      curText: '',
      showSuggsAtBos: true,
      hideSuggUnlessPartialWord: false,
      useConstraints: {},
      constraintsBySentence: 'etaoisnrhlducyfwmgpbvkzxjq',
      get curSentenceNum() {
        return (this.curText.match(/[?.!]\s/g) || []).length;
      },
      get curConstraint() {
        return {
          avoidLetter: this.useConstraints.avoidLetter ? this.constraintsBySentence[this.curSentenceNum % this.constraintsBySentence.length] : null
        };
      },
      attentionCheck: null,
      attentionCheckStats: {total: 0, passed: 0},
      tapLocations: [],
      contextSequenceNum: 0,
      lastSuggestionsFromServer: {},
      activeSuggestion: null,
      lastSpaceWasAuto: false,
      get wordCount() {
        return countWords(this.curText);
      },
      get activeSuggestionWords() {
        return this.activeSuggestion.suggestion.words.slice(this.activeSuggestion.wordIdx);
      },
      get visibleSuggestions() {
        return this.lastSuggestionsFromServer;
      },
      insertText: M.action((toInsert, charsToDelete, taps) => {
        let cursorPos = this.curText.length;
        let newCursorPos = cursorPos - charsToDelete;
        this.curText = this.curText.slice(0, newCursorPos) + toInsert;
        this.tapLocations = this.tapLocations.slice(0, newCursorPos).concat(taps || _.map(toInsert, () => null));
      }),
      tapKey: M.action(event => {
        let ac = this.validateAttnCheck(null);
        if (ac.length) return ac;

        let isNonWord = event.key.match(/\W/);
        let deleteSpace = this.lastSpaceWasAuto && isNonWord;
        let toInsert = event.key;
        if (this.curConstraint.avoidLetter === toInsert) {
          // Disallow insertion of the avoid-letter.
          return;
        }
        let taps = [{x: event.x, y: event.y}];
        let autoSpace = isNonWord && event.key !== " " && event.key !== "'" && event.key !== '-';
        if (autoSpace) {
          toInsert += " ";
          taps.push({});
        }
        this.insertText(toInsert, deleteSpace ? 1 : 0, taps);
        this.lastSpaceWasAuto = autoSpace;
        this.activeSuggestion = null;
        return [this.changedMsg()];
      }),
      tapBackspace: M.action(() => {
        /* Ignore the attention check, don't count this for or against. */
        this.insertText('', 1);
        this.lastSpaceWasAuto = false;
        this.activeSuggestion = null;
        return [this.changedMsg()];
      }),
      handleTapSuggestion: M.action((slot, which) => {
        // let ac = this.validateAttnCheck(slot);
        // if (ac.length) return ac;

       let wordToInsert = this.visibleSuggestions[which][slot];
       let {curWord} = this.getSuggestionContext();
        let charsToDelete = curWord.length;
        let isNonWord = wordToInsert.match(/^\W$/);
        let deleteSpace = this.lastSpaceWasAuto && isNonWord;
        if (deleteSpace) {
          charsToDelete++;
        }
        this.insertText(wordToInsert + ' ', charsToDelete, null);
        this.lastSpaceWasAuto = true;
        return [this.changedMsg()];
      }),

      handleSelectAlternative: M.action(event => {
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
        return [this.changedMsg()];
      }),

      updateSuggestions: M.action(event => {
        let {msg} = event;
        // Only update suggestions if the data is valid.
        if (msg.request_id !== this.contextSequenceNum) {
          // Only warn if we're more than one context behind.
          if (this.contextSequenceNum - msg.request_id > 1) {
            console.warn("Discarding outdated suggestions", msg.request_id, this.contextSequenceNum);
          }
          return;
        }
        this.lastSuggestionsFromServer = msg;
      }),
    });
  }

  changedMsg() {
    this.contextSequenceNum++;

    // Update attn check
    let rng = seedrandom(this.curText + this.contextSequenceNum);
    if (this.condition.useAttentionCheck && rng() < .1) {

      let acWord;
      if (this.curText.slice(-1) === ' ') {
        // Full-word suggestion -> put the AC anywhere.
        acWord = Math.floor(rng() * 4);
      } else {
        // partial-word -- assume they're not looking at the continuations.
        acWord = 0;
      }
      let acSlot = Math.floor(rng() * 3);
      if (this.activeSuggestion !== null && this.activeSuggestion.slot === acSlot) {
        acSlot = (acSlot + 1) % 3;
      }
      this.attentionCheck = {slot: acSlot, word: acWord};
    } else {
      this.attentionCheck = null;
    }
    return {type: 'suggestion_context_changed'};
  }

  getSuggestionContext() {
    let sofar = this.curText, cursorPos = sofar.length;
    let lastSpaceIdx = sofar.search(/\s\S*$/);
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
      constraints: this.curConstraint
    };
    if (this.activeSuggestion) {
      result.promise = {
        slot: this.activeSuggestion.slot,
        words: this.activeSuggestionWords
      };
    }
    return result;
  }

  validateAttnCheck(slot) {
    if (this.attentionCheck !== null && this.visibleSuggestions[this.attentionCheck.slot].attentionCheck) {
      this.attentionCheckStats.total++;
      if (this.attentionCheck.slot === slot) {
        this.attentionCheck = null;
        this.attentionCheckStats.passed++;
        return [{type: 'passedAttnCheck'}];
      } else {
        // Failed attn check.
        this.attentionCheck = null;
        if (false) {
          // Delete stuff...
          this.insertText('', Math.min(this.curText.length, 5));
          this.activeSuggestion = null;
          this.lastSpaceWasAuto = false;
        }
        return [{type: 'failedAttnCheck', stats: this.attentionCheckStats}];
      }
    }
    return [];
  }

  handleEvent = (event) => {
    switch (event.type) {
    case 'tapKey':
      return this.tapKey(event);
    case 'tapBackspace':
      return this.tapBackspace();
    case 'receivedSuggestions':
      return this.updateSuggestions(event);
    case 'tapSuggestion':
      return this.handleTapSuggestion(event.slot, event.which);
    case 'selectAlternative':
      return this.handleSelectAlternative(event);
    default:
    }
  };
}
