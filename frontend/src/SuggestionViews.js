import React, { Component } from "react";
import { observer, inject } from "mobx-react";
import classNames from "classnames";

class Suggestion extends Component {
  render() {
    let {
      onTap,
      word,
      preview,
      isValid,
      meta,
      beforeText,
      highlightChars,
    } = this.props;
    let highlighted = "";
    if (highlightChars) {
      highlighted = word.slice(0, highlightChars);
      word = word.slice(highlightChars);
    }
    let classes = {
      invalid: !isValid,
      bos: isValid && (meta || {}).bos,
      synonym: (meta || {}).type === 'alt'
    };
    if (!!highlightChars) {
      if (highlightChars % 2) classes.hasHighlightOdd = true;
      else classes.hasHighlightEven = true;
    }
    return (
      <div
        className={classNames("Suggestion", classes)}
        onTouchStart={isValid ? onTap : null}
        onTouchEnd={evt => {
          evt.preventDefault();
        }}
      >
        <span className="word">
          <span className="beforeText">
            {beforeText}
          </span>
          <span className="highlighted">
            {highlighted}
          </span>
          {word}
        </span>
        <span className="preview">
          {preview.join(" ")}
        </span>
      </div>
    );
  }
}

export const SuggestionsBar = inject("state", "dispatch")(
  observer(
    class SuggestionsBar extends Component {
      render() {
        const {
          dispatch,
          suggestions,
          which,
          beforeText,
          showPhrase,
        } = this.props;
        let indexedSuggs = suggestions.map((sugg, i) => ({...sugg, idx: i}));
        let sugReorder = [4, 3, 5, 1, 0, 2].map(idx => (indexedSuggs[idx] || {idx, words: []}));
        // [

        //   ...indexedSuggs.slice(3),
        //   indexedSuggs[1], indexedSuggs[0], indexedSuggs[2]]
        return (
          <div className={"SuggestionsBar " + which}>
            {sugReorder.map(sugg =>
              <Suggestion
                key={sugg.idx}
                onTap={evt => {
                  dispatch({ type: "tapSuggestion", slot: sugg.idx, which });
                  evt.preventDefault();
                  evt.stopPropagation();
                }}
                word={sugg.words[0]}
                beforeText={beforeText || ""}
                preview={showPhrase ? sugg.words.slice(1) : []}
                highlightChars={sugg.highlightChars}
                isValid={true}
                meta={sugg.meta}
              />,
            )}
          </div>
        );
      }
    },
  ),
);

export const AlternativesBar = inject("state", "dispatch")(
  observer(
    class AlternativesBar extends Component {
      render() {
        const { state } = this.props;
        let expState = state.experimentState;
        let recs = expState.visibleSuggestions;
        let heldCluster = 2;
        let selectedIdx = 9;
        let clusters = recs.clusters || [];
        let suggOffset = idx => Math.floor(idx * state.phoneSize.width / 3);
        let suggWidth = Math.floor(state.phoneSize.width / 3);
        return (
          <div className="SuggestionsContainer">
            {heldCluster &&
              <div
                className="Overlay"
                style={{ left: suggOffset(heldCluster), width: suggWidth }}
              >
                {(clusters[heldCluster] || [])
                  .reverse()
                  .map(([word, meta], wordIdx) =>
                    <span
                      key={word}
                      className={classNames(
                        wordIdx === selectedIdx && "selected",
                      )}
                    >
                      {word}
                    </span>,
                  )}
                <div className="shiftSpot" />
              </div>}
            <div className="SuggestionsBar">
              {clusters.slice(0, 3).map((cluster, clusterIdx) =>
                <div className="Suggestion" key={clusterIdx}>
                  <span className="word">
                    {cluster[0][0]}
                  </span>
                  <span className="preview" />
                </div>,
              )}
            </div>
          </div>
        );
      }
    },
  ),
);
