import M from 'mobx';
import _ from 'lodash';
import countWords from './CountWords';

/**** Main state store!

This represents all the state, UI state included. The entire UI should be a function of this, with perhaps a few pragmatic exceptions (none yet - 2016-12-17).

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
      useConstraints: {},
      constraintsBySentence: 'etaoisnrhlducyfwmgpbvkzxjq',
      get curSentenceNum() {
        return (this.curText.match(/\.\s/g) || []).length;
      },
      get curConstraint() {
        return {
          avoidLetter: this.useConstraints.avoidLetter ? this.constraintsBySentence[this.curSentenceNum % this.constraintsBySentence.length] : null
        };
      },
      tapLocations: [],
      contextSequenceNum: 0,
      lastSuggestionsFromServer: [],
      activeSuggestion: null,
      lastSpaceWasAuto: false,
      get wordCount() {
        return countWords(this.curText);
      },
      get activeSuggestionWords() {
        return this.activeSuggestion.suggestion.words.slice(this.activeSuggestion.wordIdx);
      },
      get visibleSuggestions() {
        // The visible suggestions are:
        // - The active ("promised") suggestion, in its corresponding slot.
        // - Suggestions from the server, filled in as long as they're valid.

        // Copy the suggestions so we can tweak them.
        let suggestions = M.toJS(this.lastSuggestionsFromServer).map(sugg => ({
          isValid: sugg.contextSequenceNum === this.contextSequenceNum,
          ...sugg
        }));

        if (this.activeSuggestion) {
          suggestions.splice(this.activeSuggestion.slot, 1, {
            orig: this.activeSuggestion.suggestion,
            contextSequenceNum: this.contextSequenceNum,
            words: this.activeSuggestionWords,
            isValid: true,
          });
        }
        while (suggestions.length < 3) {
          suggestions.push({
            words: [''],
            isValid: false
          });
        }
        return suggestions.slice(0, 3);
      },
      insertText: M.action((toInsert, charsToDelete, taps) => {
        let cursorPos = this.curText.length;
        let newCursorPos = cursorPos - charsToDelete;
        this.curText = this.curText.slice(0, newCursorPos) + toInsert;
        this.tapLocations = this.tapLocations.slice(0, newCursorPos).concat(taps || _.map(toInsert, () => null));
      }),
      tapKey: M.action(event => {
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
      }),
      tapBackspace: M.action(() => {
        this.insertText('', 1);
        this.lastSpaceWasAuto = false;
        this.activeSuggestion = null;
      }),
      insertSuggestion: M.action(slot => {
        let wordToInsert = null;
        let tappedSuggestion = this.visibleSuggestions[slot];
        if (this.activeSuggestion !== null && this.activeSuggestion.slot === slot) {
          // Continuing a previous suggestion.
          wordToInsert = this.activeSuggestionWords[0];
          if (this.activeSuggestionWords.length > 1) {
            // Words remain, advance the active suggestion.
            this.activeSuggestion.wordIdx++;
          } else {
            // No more words in the active suggestion, go back to normal.
            this.activeSuggestion = null;
          }
        } else if (tappedSuggestion.contextSequenceNum === this.contextSequenceNum) {
          wordToInsert = tappedSuggestion.words[0];
          if (tappedSuggestion.words.length > 1) {
            this.activeSuggestion = {
              suggestion: tappedSuggestion,
              slot: slot,
              wordIdx: 1
            };
          } else {
            this.activeSuggestion = null;
          }
        } else {
          // Invalid suggestion, ignore it.
          return;
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

        this.lastSuggestionsFromServer = msg.next_word.map(sugg => ({
          orig: sugg,
          contextSequenceNum: msg.request_id,
          words: sugg.one_word.words.concat(sugg.continuation.length ? sugg.continuation[0].words : []),
          meta: sugg.meta,
        }));
      }),
    });

    this.disposers = [];
    // Keep a running sequence of contexts.
    // This works because every context change also changes curText.
    this.disposers.push(M.observe(this, 'curText', () => {
      this.contextSequenceNum++;
    }));
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

  handleEvent = (event) => {
    switch (event.type) {
    case 'tapKey':
      this.tapKey(event);
      break;
    case 'tapBackspace':
      this.tapBackspace();
      break;
    case 'receivedSuggestions':
      this.updateSuggestions(event);
      break;
    case 'tapSuggestion':
      this.insertSuggestion(event.slot);
      break;
    default:
    }
  };

  dispose() {
    this.disposers.forEach(x => {x()});
  }
}
