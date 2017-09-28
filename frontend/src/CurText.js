import React, { Component } from "react";
import { observer, inject } from "mobx-react";
import { SuggestionsBar } from "./SuggestionViews";
import classNames from "classnames";

export const CurText = inject("spying", "state", "dispatch")(
  observer(
    class CurText extends Component {
      scrollIntoViewTimeout = null;

      scrollCursorIntoView() {
        if (this.props.spying) return;
        if (this.scrollIntoViewTimeout !== null) {
          clearTimeout(this.scrollIntoViewTimeout);
        }
        this.scrollIntoViewTimeout = setTimeout(() => {
          this.cursor.scrollIntoView();
        }, 50);
      }

      componentDidMount() {
        this.scrollCursorIntoView();
      }

      componentDidUpdate() {
        this.scrollCursorIntoView();
      }

      render() {
        let { text, replacementRange, state, dispatch } = this.props;
        let { experimentState } = state;
        let { deleting } = experimentState;
        let afterCursor = "";

        if (deleting) {
          replacementRange = null;
          afterCursor = text.slice(deleting.liveChars);
          text = text.slice(0, deleting.liveChars);
        }
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
                className={classNames("Cursor", deleting && "deleting")}
                ref={elt => {
                  this.cursor = elt;
                }}
              />
              <span className="afterCursor">
                {afterCursor}
              </span>
            </span>
          </div>
        );
      }
    },
  ),
);
