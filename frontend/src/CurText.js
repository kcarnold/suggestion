import React, { Component } from "react";
import { observer, inject } from "mobx-react";
import { SuggestionsBar } from "./SuggestionViews";

export const CurText = inject("spying", "state", "dispatch")(
  observer(
    class CurText extends Component {
      componentDidMount() {
        if (!this.props.spying) {
          this.cursor.scrollIntoView();
        }
      }

      componentDidUpdate() {
        if (!this.props.spying) {
          this.cursor.scrollIntoView();
        }
      }

      render() {
        let { text, replacementRange, state, dispatch } = this.props;
        let { experimentState } = state;

        if (!replacementRange) {
          replacementRange = [0, 0];
        }
        let [hiStart, hiEnd] = replacementRange;
        return (
          <div
            className="CurText"
            onTouchEnd={evt => {
              dispatch({ type: "tapText" });
            }}
          >
            <span>
              <span>
                {text.slice(0, hiStart)}
              </span>
              <span className="replaceHighlight">
                {text.slice(hiStart, hiEnd)}
              </span>
              <span>
                {text.slice(hiEnd)}
              </span>
              <span
                className="Cursor"
                ref={elt => {
                  this.cursor = elt;
                }}
              />
            </span>
          </div>
        );
      }
    },
  ),
);
