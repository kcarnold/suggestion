import React, { Component } from "react";
import { ControlledInput, ControlledStarRating } from "./ControlledInputs";
import { observer, inject } from "mobx-react";
import { NextBtn } from "./BaseViews";

function TextResponse({ basename, question }) {
  return (
    <ControlledInput
      name={`${basename}-${question.name}`}
      {...question.flags || {}}
    />
  );
}

function StarRating({ basename, question }) {
  let { name } = question;
  return <ControlledStarRating name={`${basename}-${name}`} />;
}

export const OptionsResponse = inject("dispatch", "state")(
  observer(function OptionsResponse({ state, dispatch, basename, question }) {
    let name = `${basename}-${question.name}`;
    let choice = state.controlledInputs.get(name) || "";
    function change(newVal) {
      dispatch({ type: "controlledInputChanged", name, value: newVal });
    }
    return (
      <div>
        {question.options.map(option =>
          <label key={option}
            style={{
              background: "#f0f0f0",
              display: "block",
              margin: "3px 0",
              padding: "10px 3px",
              width: "100%",
            }}
          >
            <input
              type="radio"
              checked={choice === option}
              onChange={() => change(option)}
            />
            <span style={{ width: "100%" }}>
              {option}
            </span>
          </label>,
        )}
      </div>
    );
  }),
);

const responseTypes = {
  starRating: StarRating,
  text: TextResponse,
  options: OptionsResponse,
};

function Question({ basename, question }) {
  console.assert(question.responseType in responseTypes);
  let responseType = responseTypes[question.responseType];
  return (
    <div
      className="Question"
      style={{
        margin: "5px",
        borderTop: "3px solid #aaa",
        padding: "5px",
      }}
    >
      <div className="QText">
        {question.text}
      </div>
      {React.createElement(responseType, { basename, question })}
    </div>
  );
}

export const PostTaskSurvey = inject("state")(
  observer(({ state }) => {
    let basename = `postTask-${state.block}`;

    let questions = [
      {
        text:
          "Now that you've had a chance to write about it, how many stars would you give your experience at this restaurant?",
        responseType: "starRating",
        name: "stars",
      },
      {
        text: "How would you describe your thought process while writing?",
        responseType: "text",
        name: "thoughtProcess",
        flags: { multiline: true },
      },
      {
        text:
          "How would you describe the shortcuts that the keyboard gave -- what they were and how you used them (or didn't use them)?",
        responseType: "text",
        name: "shortcuts",
        flags: { multiline: true },
      },

      {
        text:
          "Compared with the experience you were writing about, the shortcuts that the keyboard gave were usually...",
        responseType: "options",
        name: "sentimentManipCheck",
        options: ["More negative", "More positive", "Mixed", "Neutral"],
      },
      {
        text: "Did you experience any technical difficulties that you haven't reported already?",
        responseType: 'text',
        name: 'techDiff',
        flags: {multiline: true},
      },
      {
        text: "Any other comments? (There will be more surveys before the end of the experiment.)",
        responseType: 'text',
        name: 'other',
        flags: {multiline: true},
      }
    ];

    return (
      <div className="Survey">
        <h1>After-Writing Survey</h1>

        {questions.map(question => {
          return (
            <Question
              key={question.name}
              basename={basename}
              question={question}
            />
          );
        })}

        <NextBtn />
      </div>
    );
  }),
);
