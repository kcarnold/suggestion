import React from "react";
import { observer, inject } from "mobx-react";

function advance(state, dispatch) {
  dispatch({ type: "next" });
}

export const NextBtn = inject("dispatch", "state")(
  observer(props =>
    <button
      onClick={() => {
        if (!props.confirm || window.confirm("Are you sure?")) {
          advance(props.state, props.dispatch);
        }
      }}
      disabled={props.disabled}
    >
      {props.children || "Next"}
    </button>,
  ),
);
