import React, { Component } from "react";
import { observer, inject } from "mobx-react";
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
          this.cursor && this.cursor.scrollIntoView();
        }, 50);
      }

      componentDidMount() {
        this.scrollCursorIntoView();
      }

      componentDidUpdate() {
        this.scrollCursorIntoView();
      }

      componentWillUnmount() {
        if (this.scrollIntoViewTimeout !== null) {
          clearTimeout(this.scrollIntoViewTimeout);
          this.scrollIntoViewTimeout = null;
        }
      }

      render() {
        let { text, replacementRange, state, dispatch } = this.props;
        let { experimentState } = state;
        let { electricDeleteLiveChars } = experimentState;
        let afterCursor = "";

        if (electricDeleteLiveChars) {
          replacementRange = null;
          afterCursor = text.slice(electricDeleteLiveChars);
          text = text.slice(0, electricDeleteLiveChars);
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
                className={classNames("Cursor", electricDeleteLiveChars && "deleting")}
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
