import React, { Component } from "react";
import { ControlledInput, ControlledStarRating } from "./ControlledInputs";
import { observer, inject } from "mobx-react";
import { NextBtn } from "./BaseViews";

const miscQuestions = [
  {
    text:
      "Did you experience any technical difficulties that you haven't reported already?",
    responseType: "text",
    name: "techDiff",
    flags: { multiline: true },
  },
  {
    text:
      "Any other comments? (There will be more surveys before the end of the experiment.)",
    responseType: "text",
    name: "other",
    flags: { multiline: true },
  },
];

const postTaskQuestions = [
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
].concat(miscQuestions);

function likert(name, text, degrees, labels) {
  let options = [];
  for (let i = 0; i < degrees; i++) {
    options.push("");
  }
  options[0] = labels[0];
  options[degrees - 1] = labels[1];
  return {
    text,
    name,
    responseType: "likert",
    options,
  };
}

const tlxQuestions = [
  likert("mental", "Mental Demand: How mentally demanding was the task?", 7, [
    "Very low",
    "Very high",
  ]),
  likert(
    "physical",
    "Physical Demand: How physically demanding was the task?",
    7,
    ["Very low", "Very high"],
  ),
  likert(
    "temporal",
    "Temporal Demand: How hurried or rushed was the pace of the task?",
    7,
    ["Very low", "Very high"],
  ),
  likert(
    "performance",
    "Performance: How successful were you in accomplishing what you were asked to do?",
    7,
    ["Perfect :)", "Failure :("],
  ),
  likert(
    "effort",
    "Effort: How hard did you have to work to accomplish your level of performance?",
    7,
    ["Very low", "Very high"],
  ),
  likert(
    "frustration",
    "Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?",
    7,
    ["Very low", "Very high"],
  ),
];

const postTaskPersuade = tlxQuestions.concat(miscQuestions);

const closingSurveyQuestions = [
  {
    text: "While you were writing, did you speak or whisper what you were writing?",
    responseType: "options",
    name: "verbalized_during",
    options: ["Yes", "No"],
  },

  {
    text: "About how many online reviews (of restaurants or otherwise) have you written in the past 3 months? Don't count the reviews from this study.",
    responseType: "text",
    name: "reviewing_experience",
    flags: { type: 'number' },
  },

  {
    text:
      "Did you experience any technical difficulties that you haven't reported already?",
    responseType: "text",
    name: "techDiff",
    flags: { multiline: true },
  },
  {
    text:
      "Any other comments?",
    responseType: "text",
    name: "other",
    flags: { multiline: true },
  },
];


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
          <label
            key={option}
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

export const LikertResponse = inject("dispatch", "state")(
  observer(function LikertResponse({ state, dispatch, basename, question }) {
    let name = `${basename}-${question.name}`;
    let choice = state.controlledInputs.get(name);
    function change(newVal) {
      dispatch({ type: "controlledInputChanged", name, value: newVal });
    }
    return (
      <div
        style={{ display: "flex", flexFlow: "row nowrap", padding: "5px 0" }}
      >
        {question.options.map((label, idx) =>
          <div key={idx} style={{ textAlign: "center", flex: "1 1 0" }}>
            <label>
              <input
                type="radio"
                checked={choice === idx}
                onChange={() => change(idx)}
              />
              <br />
              <span>
                {label}&nbsp;
              </span>
            </label>
          </div>,
        )}
      </div>
    );
  }),
);

const responseTypes = {
  starRating: StarRating,
  text: TextResponse,
  options: OptionsResponse,
  likert: LikertResponse,
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
    let questions = state.isPersuade ? postTaskPersuade : postTaskQuestions;

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

export const PostExpSurvey = inject("state")(
  observer(({ state }) => {
    let basename = `postExp`;
    let questions = closingSurveyQuestions;

    return (
      <div className="Survey">
        <h1>Closing Survey</h1>

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
