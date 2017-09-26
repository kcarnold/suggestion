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

const traitItems = `
I am not easily bothered by things.
I try to avoid complex people.
I dislike myself.
I seldom feel blue.
I love to read challenging material.
I tend to vote for liberal political candidates.
I make friends easily.
I do not like art.
I do not enjoy going to art museums.
I have a vivid imagination.
I avoid difficult reading material.
I am the life of the party.
I can handle a lot of information.
I am not interested in abstract ideas.
I love to think up new ways of doing things.
I have difficulty understanding abstract ideas.
I have little to say.
I have frequent mood swings.
I enjoy hearing new ideas.
I avoid philosophical discussions.
I rarely get irritated.
I am quick to understand things.
I would describe my experiences as somewhat dull.
I believe in the importance of art.
I am skilled in handling social situations.
I tend to vote for conservative political candidates.
I am very pleased with myself.
I feel comfortable around people.
I carry the conversation to a higher level.
I don't like to draw attention to myself.
I need things explained only once.
I feel comfortable with myself.
I know how to captivate people.
I often feel blue.
I panic easily.
I don't talk a lot.
I am often down in the dumps.
I like to solve complex problems.
I keep in the background.`.trim().split(/\n/);

const personality = [
  {
    text: <p>Describe yourself as you generally are now, not as you wish to be in the future. Describe yourself as you honestly see yourself, in relation to other people you know of the same sex as you are, and roughly your same age. So that you can describe yourself in an honest manner, your responses will be kept in absolute confidence.</p>,
  },
  ...traitItems.map(item => ({
    text: item,
    name: item,
    responseType: 'likert',
    options: ["Very Inaccurate", "", "", "", "Very Accurate"]
  }))
];

const closingSurveyQuestions = [
  {
    text:
      "While you were writing, did you speak or whisper what you were writing?",
    responseType: "options",
    name: "verbalized_during",
    options: ["Yes", "No"],
  },

  {
    text:
      "About how many online reviews (of restaurants or otherwise) have you written in the past 3 months? Don't count the reviews from this study.",
    responseType: "text",
    name: "reviewing_experience",
    flags: { type: "number" },
  },

  {
    text: <h2>Demographics</h2>,
    responseType: null,
  },

  {
    text: "How old are you?",
    responseType: "text",
    name: "age",
    flags: {type: "number"}
  },

  {
    text: "What is your gender?",
    responseType: "options",
    name: "gender",
    options: ["Male", "Female", "Something else, or I'd prefer not to say"],
  },

  {
    text: "How proficient would you say you are in English?",
    responseType: "options",
    name: "english_proficiency",
    options: ["Basic", "Conversational", "Fluent", "Native or bilingual"],
  },

  ...personality,

  {
    text:
      "Did you experience any technical difficulties that you haven't reported already?",
    responseType: "text",
    name: "techDiff",
    flags: { multiline: true },
  },
  {
    text: "Any other comments?",
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
  let responseType = null;
  if (question.responseType) {
    console.assert(question.responseType in responseTypes);
    responseType = responseTypes[question.responseType];
  }
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
      {responseType && React.createElement(responseType, { basename, question })}
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

        {questions.map((question, idx) => {
          return (
            <Question
              key={question.name || idx}
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
