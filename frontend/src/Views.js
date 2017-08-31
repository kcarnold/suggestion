import React, { Component } from 'react';
import _ from 'lodash';
import {observer, inject} from 'mobx-react';
import classNames from 'classnames';
import StarRatingComponent from 'react-star-rating-component';
import {Keyboard} from './Keyboard';
import Consent from './Consent';

const hostname = window.location.host;

const surveyURLs = {
  // intro: 'SV_9GiIgGOn3Snoxwh',
  // intro: 'SV_9mGf4CUxHYIg56d', // new intro with personality
  intro: 'SV_aa5ZP3K39GLFSzr',
  postFreewrite: 'SV_0OCqAQl6o7BiidT',
  // postTask: 'SV_5yztOdf3SX8EtOl',
  // postTask: 'SV_7OPqWyf4iipivwp',
  postTask: 'SV_2tOuIDt9pSwtKIt',
  // postExp: 'SV_8HVnUso1f0DZExv',
  // postExp: 'SV_eQbXXnoiDBWeww5',
  postExp: 'SV_3K1BKZMz3O0miZT', // postExp4
}

const wordCountTarget = 75;
const askKnowWhatToSay = false;

const texts = {
  detailed: {
    overallInstructions: <span>Write the true story of your experience. Tell your reader <b>as many vivid details as you can</b>. Don't worry about <em>summarizing</em> or <em>giving recommendations</em>.</span>,
    brainstormingInstructions: <span><b>Brainstorm what you might want to talk about</b> by typing anything that comes to mind, even if it's not entirely accurate. Don't worry about grammar, coherence, accuracy, or anything else, this is just for you. <b>Have fun with it</b>, we'll write the real thing in step 2.</span>,
    revisionInstructions: <span>Okay, this time for real. Try to make it reasonably accurate and coherent.</span>,
    instructionsQuiz: 'SV_42ziiSrsZzOdBul',
  },
  detailedNoBrainstorm: {
    overallInstructions: <span>Write the true story of your experience. Tell your reader <b>as many vivid details as you can</b>. Don't worry about <em>summarizing</em> or <em>giving recommendations</em>.</span>,
    revisionInstructions: <span>Try to make it reasonably accurate and coherent.</span>,
    instructionsQuiz: 'SV_42ziiSrsZzOdBul',
  },
  funny: {
    overallInstructions: <span>Write the <b>funniest</b> review you can come up with. Have fun with it!</span>,
    brainstormingInstructions: <span><b>Brainstorm what you might want to talk about</b> by typing anything that comes to mind, even if it's not entirely accurate. Don't worry about grammar, coherence, accuracy, or anything else, this is just for you. <b>Have fun with it</b>, we'll write the real thing in step 2.</span>,
    revisionInstructions: <span>Okay, this time for real. Try to make it reasonably accurate and coherent -- but still funny!</span>,
    instructionsQuiz: null,
  },
  review: {
    overallInstructions: <span>Write a review of your experience that you'd be proud to post on a review website. Use at least {wordCountTarget} words. We'll bonus our favorite reviews!</span>,
    brainstormingInstructions: <span />,
    revisionInstructions: <span/>,
    instructionsQuiz: null,
  },
  tabooTopic: {
    overallInstructions: <span>Write a review of your experience <b>that tells us something new</b>, something that other reviews probably don't tell us. Specifically, other reviews already talk a lot about the <em>food</em>, <em>service</em>, and <em>atmosphere</em>, so try to <b>focus on other topics</b>. <br/><br/>Use at least {wordCountTarget} words. We'll bonus our favorite reviews!</span>,
    brainstormingInstructions: <span />,
    revisionInstructions: <span/>,
    instructionsQuiz: null,
  },
  sentiment: {
    overallInstructions: <span>Write a review of your experience. Include <b>both positive and negative aspects</b>. Use at least {wordCountTarget} words. We'll bonus our favorite reviews!</span>,
    brainstormingInstructions: <span />,
    revisionInstructions: <span/>,
    instructionsQuiz: null,
  },
  yelp: {
    overallInstructions: <span>
      Write a review of your experience. &ldquo;<em>The best reviews are passionate, personal, and accurate.
      They offer a rich narrative, a wealth of detail, and a helpful tip or two for other consumers.</em>&rdquo; (based on Yelp's Guidelines)...
      and please try to avoid typos. <b>We'll bonus our favorite reviews</b>!
    </span>,
    brainstormingInstructions: null,
    revisionInstructions: null,
    instructionsQuiz: null,
  }
};

const tutorialTaskDescs = {
  typeKeyboard: 'Type a few words by tapping letters on the keyboard.',
  backspace: 'Try deleting a few letters.',
  specialChars: 'Try typing some punctuation (period, comma, apostrophe, etc.)',
  tapSuggestion: 'Try tapping a box to insert the word.',
  tapPrediction: 'Try tapping a grey box to insert the word.',
  tapAlternative: "Try tapping a green box to replace the highlighted word with it."
};

class Suggestion extends Component {
  render() {
    let {onTap, word, preview, isValid, meta, beforeText, highlightChars} = this.props;
    let highlighted = '';
    if (highlightChars) {
      highlighted = word.slice(0, highlightChars);
      word = word.slice(highlightChars);
    }
    let classes = {
      invalid: !isValid,
      bos: isValid && (meta || {}).bos
    };
    if (!!highlightChars) {
      if (highlightChars % 2)
        classes.hasHighlightOdd = true;
      else
        classes.hasHighlightEven = true
    }
    return <div
      className={classNames("Suggestion", classes)}
      onTouchStart={isValid ? onTap : null}
      onTouchEnd={evt => {evt.preventDefault();}}>
      <span className="word"><span className="beforeText">{beforeText}</span><span className="highlighted">{highlighted}</span>{word}</span><span className="preview">{preview.join(' ')}</span>
    </div>;
  }
}

const SuggestionsBar = inject('state', 'dispatch')(observer(class SuggestionsBar extends Component {
  render() {
    const {dispatch, suggestions, which, beforeText} = this.props;
    return <div className={"SuggestionsBar " + which}>
      {(suggestions || []).map((sugg, i) => <Suggestion
        key={i}
        onTap={(evt) => {
          dispatch({type: 'tapSuggestion', slot: i, which});
          evt.preventDefault();
          evt.stopPropagation();
        }}
        word={sugg.words[0]}
        beforeText={beforeText || ''}
        preview={sugg.words.slice(1)}
        highlightChars={sugg.highlightChars}
        isValid={true}
        meta={null} />
      )}
    </div>;
  }
}));

const AlternativesBar = inject('state', 'dispatch')(observer(class AlternativesBar extends Component {
  render() {
    const {state, dispatch} = this.props;
    let expState = state.experimentState;
    let recs = expState.visibleSuggestions;
    let heldCluster = 2;
    let selectedIdx = 9;
    let clusters = recs.clusters || [];
    let suggOffset = (idx) => Math.floor(idx * state.phoneSize.width / 3);
    let suggWidth = Math.floor(state.phoneSize.width / 3);
    return <div className="SuggestionsContainer">
      {heldCluster && <div className="Overlay" style={{left: suggOffset(heldCluster), width: suggWidth}}>
        {(clusters[heldCluster] || []).reverse().map(([word, meta], wordIdx) => <span key={word} className={classNames(wordIdx === selectedIdx && 'selected')}>{word}</span>)}
        <div className="shiftSpot" />
      </div>}
      <div className="SuggestionsBar">
      {clusters.slice(0, 3).map((cluster, clusterIdx) =>
        <div className="Suggestion" key={clusterIdx}><span className="word">{cluster[0][0]}</span><span className="preview" /></div>)}</div>
    </div>;
  }
}));


function advance(state, dispatch) {
  dispatch({type: 'next'})
}

const NextBtn = inject('dispatch', 'state')((props) => <button onClick={() => {
  if (!props.confirm || window.confirm("Are you sure?")) {
    advance(props.state, props.dispatch);
  }
  }} disabled={props.disabled}>{props.children || "Next"}</button>);


const ControlledInput = inject('dispatch', 'state')(observer(function ControlledInput({state, dispatch, name, ...props}) {
  return <input
    name={name}
    onChange={evt => {dispatch({type: 'controlledInputChanged', name, value: evt.target.value});}}
    value={state.controlledInputs.get(name) || ''}
    {...props} />;
  }));

const ControlledStarRating = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <div>
  <StarRatingComponent
    name={name} starCount={5} value={state.controlledInputs.get(name) || 0}
    onStarClick={value => {dispatch({type: 'controlledInputChanged', name, value});}}
    renderStarIcon={(idx, value) => <i style={{fontStyle: 'normal'}}>{idx<=value ? '\u2605' : '\u2606'}</i>} />
  {state.controlledInputs.get(name)}
  </div>));


const qualtricsPrefix = 'https://harvard.az1.qualtrics.com/SE/?SID=';

const RedirectToSurvey = inject('state', 'clientId', 'clientKind', 'spying')(class RedirectToSurvey extends Component {
  getRedirectURL() {
    let afterEvent =  this.props.afterEvent || 'completeSurvey';
    let nextURL = `${window.location.protocol}//${window.location.host}/?${this.props.clientId}-${this.props.clientKind}#${afterEvent}`;
    let url = nextURL;
    if (this.props.url) {
      url = `${qualtricsPrefix}${this.props.url}&clientId=${this.props.clientId}&nextURL=${encodeURIComponent(nextURL)}`;
      if (this.props.extraParams) {
        url += '&' + _.map(this.props.extraParams, (v, k) => `${k}=${v}`).join('&');
      }
    }
    return url;
  }

  componentDidMount() {
    if (this.props.spying) return;
    // This timeout is necessary to give the current page enough time to log the event that caused this render.
    // 2 seconds is probably overdoing it, but on the safe side.
    this.timeout = setTimeout(() => {
      window.location.href = this.getRedirectURL();
      if (!this.props.url) {
        // reload to trigger the event.
        window.location.reload();
      }
    }, 2000);
  }

  componentWillUnmount() {
    // Just in case.
    clearTimeout(this.timeout);
  }

  render() {
    if (this.props.spying) {
      let url = this.getRedirectURL();
      return <div>(survey: {this.props.state.curScreen.controllerScreen}) <a href={url}>{url}</a></div>;
    }
    return <div>redirecting...</div>;
  }
});

const TutorialTodo = ({done, children}) => <div style={{color: done ? 'green' : 'red'}}>{done ? '\u2611' : '\u2610'} {children}</div>;

const CurText = inject('spying', 'state', 'dispatch')(observer(class CurText extends Component {
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
    let {text, replacementRange, state, dispatch} = this.props;
    if (!replacementRange) {
      replacementRange = [0, 0];
    }
    if (state.experimentState.attentionCheck && state.experimentState.attentionCheck.type === 'text')
      text = text + 'æ';
    let [hiStart, hiEnd] = replacementRange;
    return <div className="CurText" onTouchEnd={evt => {dispatch({type: 'tapText'});}}><span>
      <span>{text.slice(0, hiStart)}</span>
      <span className="replaceHighlight">{text.slice(hiStart, hiEnd)}</span>
      <span>{text.slice(hiEnd)}</span>
      <span className="Cursor" ref={elt => {this.cursor = elt;}}></span></span></div>;
  }
}));

export const Welcome = () => <div>
    <h1>Welcome</h1>
    <Consent />
    <p>If you consent to participate, click here: <NextBtn /></p>
    </div>;

export const  ProbablyWrongCode = () => <div>
    <p>Waiting for computer. If you're seeing this on your phone, you probably mistyped your code.</p>
  </div>;

function RestaurantPrompt({idx}) {
  return <div key={idx} className="Restaurant">{idx}.
    Name: <ControlledInput name={`restaurant${idx}`} /><br />
    About how long ago were you there, in days? <ControlledInput name={`visit${idx}`} type="number" min="0"/>
    <br />How would you rate that visit? <ControlledStarRating name={`star${idx}`} />
    {askKnowWhatToSay && <span><br/><br />On a scale of 1 to 5, do you already know what you want to say about this experience? 1="I haven't thought about it at all yet", 5="I know exactly what I want to say"<br/>
    <ControlledInput name={`knowWhat${idx}`} type="number" min="1" max="5" /></span>}
  </div>;
}

export const SelectRestaurants = inject('state')(observer(({state}) => {
  let numPlaces = state.conditions.length;
  let indices = state.conditions.map((condition, idx) => idx + 1);
  let groups = [{header: null, indices}];
  if (state.masterConfigName === 'sent4') {
    groups = [
      {header: "Above-average experiences", indices: [1, 2]},
      {header: "Below-average experiences", indices: [3, 4]}
    ];
  }
  let allFields = [];
  indices.forEach(idx => {
    ['restaurant', 'visit', 'star', 'knowWhat'].forEach(kind => {
      if (kind === 'knowWhat' && !askKnowWhatToSay) return;
      allFields.push(`${kind}${idx}`);
    });
  });
  let complete = _.every(allFields, x => state.controlledInputs.get(x))

  return <div>
    <h1>Ready to write some reviews?</h1>
    <p>Think of {numPlaces} <b>restaurants (or bars, cafes, diners, etc.)</b> you've been to recently that you <b>haven't written about before</b>.</p>
    {state.masterConfigName === 'sent4' && <p>Try to pick 2 above-average experiences and 2 below-average experiences.</p>}

    {groups.map(({header, indices: groupIndices}, groupIdx) => <div key={groupIdx} style={{borderLeft: '2px solid black', paddingLeft: '5px'}}>
      {header && <h3>{header}</h3>}
      {groupIndices.map(idx => <RestaurantPrompt  key={idx} idx={idx} />)}
    </div>)}

    {complete || <p>(The Next button will be enabled once all fields are filled out.)</p>}
    <NextBtn disabled={!complete} />
  </div>;
}));


export const Instructions = inject('state')(observer(({state}) => {
    let inExperiment = state.curScreen.screen === 'ExperimentScreen';
    let {isPrewrite} = state;
    return <div>
      <h1>Let's write about your experience at {state.curPlace.name}!</h1>
      <p>Think about your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>.</p>
      <p style={{border: '1px solid black', padding: '2px'}}>{texts[state.masterConfig.instructions].overallInstructions}</p>
      {state.prewrite &&  <p>We'll do this in <b>two steps</b>:</p>}
      {state.prewrite &&  <ol>
        <li style={{paddingBottom: '1em', color: state.passedQuiz && isPrewrite ? 'blue' : 'black'}}>{texts[state.masterConfig.instructions].brainstormingInstructions}</li>
        <li style={{color: !isPrewrite ? 'blue' : 'black'}}>{texts[state.masterConfig.instructions].revisionInstructions}</li>
      </ol>}
      {state.prewrite
        ? <p>Both steps will happen on your phone, using the keyboard you just practiced with.</p>
        : <p>{false && texts[state.masterConfig.instructions].revisionInstructions}</p>}
      <hr/>
      {state.passedQuiz || inExperiment || texts[state.masterConfig.instructions].instructionsQuiz === null
        ? <p>Use your phone to complete this step.</p>
        : <p>Your phone shows a brief quiz on these instructions. Once you've passed the quiz, look back here.</p>}
      <p>The shortcuts will be different from what you saw before.</p>
    </div>;
  }));

//     <p>{texts[state.masterConfig.instructions].overallInstructions}</p>

export const ReadyPhone = inject('state')(observer(({state}) => state.passedQuiz ? <div>
    <p>Your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b></p>
    <p>{texts[state.masterConfig.instructions].overallInstructions}</p>
    <p>{state.isPrewrite ? texts[state.masterConfig.instructions].brainstormingInstructions : texts[state.masterConfig.instructions].revisionInstructions}</p>
    {state.condition.useAttentionCheck && <p>For this study, we need to measure which parts of the screen people are paying attention to. So if you happen to notice an "æ" somewhere, tap it to acknowledge that you saw it. (Don't worry if you happen to miss a few, and sorry if it gets annoying.)</p>}
    <p>If you need a break, take it before tapping Next. Tap Next when you're ready to start.<br/><br/><NextBtn /></p></div>
    : <RedirectToSurvey url={texts[state.masterConfig.instructions].instructionsQuiz} afterEvent={'passedQuiz'} extraParams={{prewrite: state.prewrite}} />));

/*  InstructionsQuiz: inject('state')(({state}) => state.passedQuiz ? <p>You already passed the quiz the first time, just click <NextBtn /></p> : ),*/

export const RevisionComputer = inject('state')(observer(({state}) => <div>
  <p><b>Now use your phone to write about your experience at {state.curPlace.name}.</b></p>

  <p>{texts[state.masterConfig.instructions].overallInstructions}</p>
      <div>Word count: {state.experimentState.wordCount}</div>
      {state.experimentState.wordCount < wordCountTarget ? <div>Try to write {wordCountTarget} words.</div> : <div>When you're done, click here: <NextBtn /></div>}
      {state.prewrite && <div>
        <p>Here is what you wrote last time:</p>
        <div style={{whiteSpace: 'pre-line'}}>{state.experiments.get(`pre-${state.block}`).curText}</div>
      </div>}
    </div>));

const OutlineSelector = inject('state', 'dispatch')(observer(({state, dispatch}) => {
  return <div className="OutlineSelector">
    {state.prewriteLines.map((line, idx) => <span key={idx} className={classNames({cur: idx===state.curPrewriteLine})} onClick={() => dispatch({type: 'selectOutline', idx})}>{line}</span>)}
    <button onClick={() => {
      let line = prompt();
      if (line && line.length) {
        dispatch({type: "addPrewriteItem", line});
      }
    }}>Add</button>
  </div>;
}));


export const ExperimentScreen = inject('state', 'dispatch')(observer(({state, dispatch}) => {
      let {experimentState, isPractice} = state;
      let {showReplacement, showSynonyms, showPredictions} = state.experimentState;
      let beforeText = ''; // experimentState.curText.slice(0, (state.experimentState.visibleSuggestions['replacement_range'] || [0])[0]).slice(-20);
      return <div className="ExperimentScreen">
        <div className="header">
          {isPractice ? "See computer for instructions." : <span>{
            state.prewrite ? (state.isPrewrite ? "Brainstorming for your" : "Revised") : "Your"} <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>
          </span>}
          {experimentState.curConstraint.avoidLetter ? <div>This sentence cannot use the letter <b>{experimentState.curConstraint.avoidLetter}</b>.</div> : null}
          {state.condition.useAttentionCheck && <p>If you notice an æ, tap on it (or nearby, it doesn't matter). Don't worry if you happen to miss a few.</p>}
          {state.condition.useAttentionCheck && <div className={classNames("missed-attn-check", state.showAttnCheckFailedMsg ? "active" : "inactive")}>There was an æ in an area you haven't noticed yet!<br/>Look for the æ and tap it.<br/>Once you notice it yourself, these messages will stop.</div>}
          {state.condition.usePrewriteText && <OutlineSelector />}
        </div>
        <CurText text={experimentState.curText} replacementRange={showReplacement && experimentState.visibleSuggestions['replacement_range']} />
        {state.condition.alternatives ? <AlternativesBar /> : <div>
          {showSynonyms && <SuggestionsBar which="synonyms" suggestions={experimentState.visibleSuggestions['synonyms']} beforeText={beforeText} />}
          {showPredictions && <SuggestionsBar which="predictions" suggestions={experimentState.visibleSuggestions['predictions']} />}
        </div>}
        <Keyboard dispatch={dispatch} />
      </div>;
    }));

export const PracticeWord = inject('state', 'dispatch')(observer(({state, dispatch}) => {
    let allTasksDone = _.every(['typeKeyboard', 'backspace', 'specialChars', 'tapSuggestion'].map(name => state.tutorialTasks.tasks[name]));
    return <div>
      <p>For technical reasons, we have to use a special keyboard for this experiment. It will probably feel harder to type with than your ordinary keyboard, and it's missing some characters you may want to type, sorry about that.</p>
      <p>Let's get a little practice with it:</p>
      {['typeKeyboard', 'backspace', 'specialChars'].map(name => <TutorialTodo key={name} done={state.tutorialTasks.tasks[name]}>{tutorialTaskDescs[name]}</TutorialTodo>)}
      <p>Don't worry about capitalization, numbers, or anything else that isn't on the keyboard.</p>

      <p>Notice the 3 boxes above the keyboard. Each one shows a word, tap a word to insert it.</p>
      {['tapSuggestion'].map(name => <TutorialTodo key={name} done={state.tutorialTasks.tasks[name]}>{tutorialTaskDescs[name]}</TutorialTodo>)}
      {allTasksDone ? <p>When you're ready, click here to move on: <NextBtn />.</p> : <p>Complete all of the tutorial steps to move on.</p>}
    </div>;
  }));

      // <video src="demo4.mp4" controls ref={elt => {elt.playbackRate=2;}}/>

export const PracticeComputer = inject('state', 'dispatch')(observer(({state, dispatch}) => {
  let previewPhrase3 = state.experimentState.visibleSuggestions.predictions[0].words.slice(0, 3).join(' ');
    return <div className="Tutorial">
      <h1>Tutorial (part 1 of 2)</h1>

      <p>For technical reasons, we have to use a special keyboard for this experiment. It might feel harder to type with it than your ordinary keyboard, and it's missing some characters you may want to type, sorry about that.
      But it has a few special features that you get to try out!</p>

      <ul>
        <li>Try the shortcut buttons to insert words:<br/>
        <TutorialTodo done={state.tutorialTasks.tasks.tapSuggestion}>Tap one of the boxes to insert that word.</TutorialTodo>
        </li>
        <li>To help you get used to the shortcuts, if you start tapping a word that you could use a shortcut for, the shortcut will light up.</li>
        <li>Each shortcut button shows a preview of the words that it will insert if you tap it repeatedly. For example, if you triple-tap the box on the left, it will insert &ldquo;<tt>{previewPhrase3}</tt>&rdquo;.
        <TutorialTodo done={state.tutorialTasks.tasks.doubleTap}>Try a <b>double-tap</b> to insert two words.</TutorialTodo>
        </li>
      </ul>

      <p>Occasionally, double-tapping may cause your phone to zoom its screen. Unfortunately there's not much we can do about that. If that happens, try double-tapping on an empty area, or reload the page (you won't lose your work).</p>

      Of course, the keys also work. To keep things simple, there's no upper-case, and just a limited amount of punctuation.
      {['typeKeyboard', 'backspace', 'specialChars'].map(name => <TutorialTodo key={name} done={state.tutorialTasks.tasks[name]}>{tutorialTaskDescs[name]}</TutorialTodo>)}

      <p>Don't worry about capitalization, numbers, or anything else that isn't on the keyboard.</p>
      {_.every(['typeKeyboard', 'backspace', 'specialChars', 'tapSuggestion', 'doubleTap'].map(name => state.tutorialTasks.tasks[name])) ? <p>
        Ready for the second part of the tutorial? Click <NextBtn />.</p> : <p>Complete all of the steps above to move on.</p>}
    </div>;
  }));

export const TutorialInstructions = inject('state', 'dispatch')(observer(({state, dispatch}) => {
  let { curText } = state.experimentState;
  let hasPeriod = curText.indexOf('.') !== -1;
  let hasSentence = hasPeriod && curText.length > 10;
  return <div className="Tutorial">
    <h1>Tutorial (part 2 of 2)</h1>

    <p>Ok now to try actually writing something. We're still in the tutorial, so it'll be something really simple.</p>

    <p><b>Tutorial task</b>:</p>

    <ul>
      <li>Think of a residence that you know well -- where you live now, where you grew up, etc.</li>
      <li>Imagine you're writing a description of it for a site like Airbnb or Craigslist. (Please don't include any information that would identify you.)</li>
      <li><b>Write a sentence about the interior of the residence.</b></li>
      <li>Try to write it using <b>as few taps</b> as possible. Don't worry about capitalization, numbers, or anything else that isn't on the keyboard.</li>
    </ul>

    {!hasSentence && <p><b>Please type a complete sentence before moving on.</b></p>}

    <p>Once you've typed a complete sentence (a few words and a period), click <NextBtn />.</p>
  </div>;
}));

export const PracticeAlternativesInstructions = inject('state', 'dispatch')(observer(({state, dispatch}) => {
    return <div>
      <p>Now we've changed the keyboard a little.</p>
      <ul>
        <li>After you type a word, it will be highlighted in green.</li>
        <li>Green boxes will show alternatives to that word.</li>
        <li>Tap any of the alternatives to use it <em>instead of</em> the green word.</li>
      </ul>

      <p><b>Practice task</b>: Write the same sentence again, but try out some of the alternatives.</p>

      {_.every(['tapAlternative'].map(name => state.tutorialTasks.tasks[name])) ? <p>
        After you've written your sentence, click here: <NextBtn />.</p> : <p>Make sure you try out the alternatives :)</p>}
    </div>;
  }));




export const TimesUpPhone = () => <div>Time is up. Follow the instructions on your computer.</div>;

export const EditScreen = inject('state', 'dispatch')(observer(({state, dispatch}) => <div className="EditPage">
    <div style={{backgroundColor: '#ccc', color: 'black'}}>
      Now, edit what you wrote to make it better.
    </div>
    <textarea value={state.curEditText} onChange={evt => {dispatch({type: 'controlledInputChanged', name: state.curEditTextName, value: evt.target.value});}} />;
  </div>));

export const ListWords = inject('state', 'dispatch')(observer(({state, dispatch}) => <div className="ListWords">
    <p>Think about your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>.</p>
    <p style={{border: '1px solid black', padding: '2px', maxWidth: '50%'}}>{texts[state.masterConfig.instructions].overallInstructions}</p>

    <p style={{paddingTop: "10px"}}><b>Write 5-10 words or phrases, one per line, that come to mind as you think about your experience.</b></p>
    <p>Consider the food, drinks, service, ambiance, location, etc.</p>

    <textarea rows={12} value={state.prewriteText}
      placeholder="One word or phrase per line"
      onChange={evt => {dispatch({type: 'prewriteTextChanged', value: evt.target.value});}} />
    <NextBtn/>
  </div>));


export const IntroSurvey = () => <RedirectToSurvey url={surveyURLs.intro} />;
export const PostFreewriteSurvey = () => <RedirectToSurvey url={surveyURLs.postFreewrite} />;
export const PostTaskSurvey = inject('state')(({state}) => <RedirectToSurvey url={surveyURLs.postTask} extraParams={{prewrite: state.prewrite}} />);
export const PostExpSurvey = () => <RedirectToSurvey url={surveyURLs.postExp} />;
export const Done = inject('clientId', 'state')(({clientId, state}) => <div>Thanks! Your code is <tt style={{fontSize: '20pt'}}>{clientId}</tt><br/><br />
  {state.isHDSL && <p>Your participation has been logged. Expect to receive a gift certificate by email in the next few days. Thanks!
    <img src={`https://harvarddecisionlab.sona-systems.com/webstudy_credit.aspx?experiment_id=440&credit_token=2093214a21504aae88bd36405e5a4e08&survey_code=${state.participantCode}`} alt="" /></p>}
  <p>In case you want them, here's what you wrote.</p>
      {state.places.map(({name}, idx) => <div key={idx}>
      <h1>{idx+1}: {name}</h1>
      <div style={{border: '1px solid black', margin: '5px'}}>{state.experiments.get(`final-${idx}`).curText}</div>
    </div>)}

  </div>);
export const LookAtPhone = inject('clientId')(({clientId}) => <div><p>Complete this step on your phone.</p> If you need it, your phone code is <tt>{clientId}-p</tt>.</div>);
export const LookAtComputer = inject('clientId')(({clientId}) => <div><p>Complete this step on your computer.</p> If you need it, your computer code is <tt>{clientId}-c</tt>.</div>);

export const SetupPairingComputer = inject('clientId')(({clientId}) => {
    let url = `http://${hostname}/?${clientId}-p`;
    return <div>
    <p>You will need two devices to complete this study: a <b>laptop/desktop computer</b> (you could use a tablet but we haven't tested it), and a <b>smartphone</b> with a web browser and WiFi (we will not be responsible for any data charges).</p>

    <div>How to pair your phone (they're all the same, pick the easiest one for you):</div>
    <ul>
      <li>On your phone's web browser, go to <tt>{hostname}</tt> and enter <tt>{clientId}-p</tt>.</li>
      <li>Send this link to yourself: <input readOnly={true} style={{fontFamily: 'monospace', width: '25em'}} value={url} /></li>
      <li>Scan this:<br/><img src={"https://zxing.org/w/chart?cht=qr&chs=350x350&chld=L&choe=UTF-8&chl=" + encodeURIComponent(url)} alt=""/></li>
    </ul>
    <p>Once your phone is paired, there will be a button on that page to continue.</p>
  </div>;
  });
export const SetupPairingPhone = () => <div>Successfully paired! <NextBtn /></div>;

export const ShowReviews = inject('state')(observer(({state}) => <div>
    <p>Here's what you wrote:</p>
    {state.places.map(({name}, idx) => <div key={idx}>
      <h1>{idx+1}: {name}</h1>
      <div style={{border: '1px solid black', margin: '5px'}}>{state.experiments.get(`final-${idx}`).curText}</div>
    </div>)}
  </div>));
