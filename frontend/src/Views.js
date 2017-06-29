import React, { Component } from 'react';
import _ from 'lodash';
import {observer, inject} from 'mobx-react';
import classNames from 'classnames';
import StarRatingComponent from 'react-star-rating-component';
import {Keyboard} from './Keyboard';
import Consent from './Consent';

const hostname = window.location.host;

const surveyURLs = {
  intro: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_9GiIgGOn3Snoxwh',
  postFreewrite: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_0OCqAQl6o7BiidT',
  // postTask: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_5yztOdf3SX8EtOl',
  postTask: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_7OPqWyf4iipivwp',
  // postExp: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_8HVnUso1f0DZExv',
  postExp: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_eQbXXnoiDBWeww5',
}

const wordCountTarget = 100;

const texts = {
  detailed: {
    overallInstructions: <span>Write the true story of your experience. Tell your reader <b>as many vivid details as you can</b>. Don’t worry about <em>summarizing</em> or <em>giving recommendations</em>.</span>,
    brainstormingInstructions: <span><b>Brainstorm what you might want to talk about</b> by typing anything that comes to mind, even if it's not entirely accurate. Don't worry about grammar, coherence, accuracy, or anything else, this is just for you. <b>Have fun with it</b>, we'll write the real thing in step 2.</span>,
    revisionInstructions: <span>Okay, this time for real. Try to make it reasonably accurate and coherent.</span>,
    instructionsQuiz: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_42ziiSrsZzOdBul',
  },
  detailedNoBrainstorm: {
    overallInstructions: <span>Write the true story of your experience. Tell your reader <b>as many vivid details as you can</b>. Don’t worry about <em>summarizing</em> or <em>giving recommendations</em>.</span>,
    revisionInstructions: <span>Try to make it reasonably accurate and coherent.</span>,
    instructionsQuiz: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_42ziiSrsZzOdBul',
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
    overallInstructions: <div>
      <p>Write a review of your experience.</p>
      <blockquote>The best reviews are passionate, personal, and accurate. They offer a rich narrative, a wealth of detail, and a helpful tip or two for other consumers.</blockquote>
      <p>(based on Yelp's Guidelines)</p>
    </div>,
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
};

class Suggestion extends Component {
  render() {
    let {onTap, word, preview, isValid, meta} = this.props;
    return <div
      className={classNames("Suggestion", {invalid: !isValid, bos: isValid && (meta || {}).bos})}
      onTouchStart={isValid ? onTap : null}
      onTouchEnd={evt => {evt.preventDefault();}}>
      <span className="word">{word}</span><span className="preview">{preview.join(' ')}</span>
    </div>;
  }
}

const SuggestionsBar = inject('state', 'dispatch')(observer(class SuggestionsBar extends Component {
  render() {
    const {state, dispatch} = this.props;
    let expState = state.experimentState;
    let {showPhrase} = expState.condition;
    return <div className="SuggestionsBar">
      {expState.visibleSuggestions.map((sugg, i) => <Suggestion
        key={i}
        onTap={(evt) => {
          dispatch({type: 'tapSuggestion', slot: i});
          evt.preventDefault();
          evt.stopPropagation();
        }}
        word={sugg.words[0]}
        preview={showPhrase ? sugg.words.slice(1) : []}
        isValid={sugg.isValid}
        meta={sugg.meta} />
      )}
    </div>;
  }
}));

function advance(state, dispatch) {
  dispatch({type: 'next'})
}

const NextBtn = inject('dispatch', 'state')((props) => <button onClick={() => {
  if (!props.confirm || confirm("Are you sure?")) {
    advance(props.state, props.dispatch);
  }
  }} disabled={props.disabled}>{props.children || "Next"}</button>);


const ControlledInput = inject('dispatch', 'state')(observer(({state, dispatch, name, ...props}) => <input
  onChange={evt => {dispatch({type: 'controlledInputChanged', name, value: evt.target.value});}}
  value={state.controlledInputs.get(name) || ''}
  {...props} />));

const ControlledStarRating = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <StarRatingComponent
  name={name} starCount={5} value={state.controlledInputs.get(name) || 0}
  onStarClick={value => {dispatch({type: 'controlledInputChanged', name, value});}}
  renderStarIcon={(idx, value) => <i style={{fontStyle: 'normal'}}>{idx<=value ? '\u2605' : '\u2606'}</i>} />));



const RedirectToSurvey = inject('state', 'clientId', 'clientKind', 'spying')(class RedirectToSurvey extends Component {
  getRedirectURL() {
    let afterEvent =  this.props.afterEvent || 'completeSurvey';
    let nextURL = `${window.location.protocol}//${window.location.host}/?${this.props.clientId}-${this.props.clientKind}#${afterEvent}`;
    let url = nextURL;
    if (this.props.url) {
      url = `${this.props.url}&clientId=${this.props.clientId}&nextURL=${encodeURIComponent(nextURL)}`;
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

const CurText = inject('spying')(observer(class CurText extends Component {
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
    return <div className="CurText"><span>{this.props.text}<span className="Cursor" ref={elt => {this.cursor = elt;}}></span></span></div>;
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

export const SelectRestaurants = inject('state')(observer(({state}) => {
  let numPlaces = state.conditions.length;
  let indices = state.conditions.map((condition, idx) => idx + 1)
  let allFields = [];
  indices.forEach(idx => {
    ['restaurant', 'visit', 'star', 'knowWhat'].forEach(kind => {
      allFields.push(`${kind}${idx}`);
    });
  });
  let complete = _.every(allFields, x => state.controlledInputs.get(x))

  return <div>
    <p>Think of {numPlaces} <b>restaurants (or bars, cafes, diners, etc.)</b> you've been to recently that you <b>haven't written about before</b>.</p>
    {state.masterConfigName === 'sent4' && <p>Try to pick 2 above-average experiences and 2 below-average experiences.</p>}
    {indices.map(idx => <div key={idx} className="Restaurant">{idx}.
      Name: <ControlledInput name={`restaurant${idx}`} /><br />
      About how long ago were you there, in days? <ControlledInput name={`visit${idx}`} type="number" min="0"/>
      <br />How would you rate that visit? <ControlledStarRating name={`star${idx}`} />
      <br/><br />On a scale of 1 to 5, do you already know what you want to say about this experience? 1="I haven't thought about it at all yet", 5="I know exactly what I want to say"<br/>
      <ControlledInput name={`knowWhat${idx}`} type="number" min="1" max="5" />
    </div>)}
    <p>(The Next button will be enabled once all fields are filled out.)</p>
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
    </div>;
  }));

//     <p>{texts[state.masterConfig.instructions].overallInstructions}</p>

export const ReadyPhone = inject('state')(observer(({state}) => state.passedQuiz ? <div>
    <p>Your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b></p>
    <p>{texts[state.masterConfig.instructions].overallInstructions}</p>
    <p>{state.isPrewrite ? texts[state.masterConfig.instructions].brainstormingInstructions : texts[state.masterConfig.instructions].revisionInstructions}</p>
    <p><b>Note</b>: We've changed how the keyboard picks words or phrases to show.</p>
    <p>Tap Next when you're ready to start. (If you need a break, take it before tapping Next.)<br/><br/><NextBtn /></p></div>
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
      let {experimentState} = state;
      return <div className="ExperimentScreen">
        <div className="header">
          {state.prewrite ? (state.isPrewrite ? "Brainstorming for your" : "Revised") : "Your"} <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>
          {experimentState.curConstraint.avoidLetter ? <div>This sentence cannot use the letter <b>{experimentState.curConstraint.avoidLetter}</b>.</div> : null}
          <p>If "æ" appears anywhere in one of the boxes above the keyboard, tap the box. (Don't worry if you miss a few.)</p>

          <p>æ shown {experimentState.attentionCheckStats.total} times, noticed {experimentState.attentionCheckStats.passed} times.</p>
          {state.condition.usePrewriteText && <OutlineSelector />}
        </div>
        <CurText text={experimentState.curText} />
        <SuggestionsBar />
        <Keyboard dispatch={dispatch} />
      </div>;
    }));

export const PracticePhone = inject('state', 'dispatch')(observer(({state, dispatch}) => {
    let {experimentState} = state;
    return <div className="ExperimentScreen">
      <div className="header">See computer for instructions.
          {experimentState.curConstraint.avoidLetter ? <div>This sentence cannot use the letter <b>{experimentState.curConstraint.avoidLetter}</b>.</div> : null}
      </div>
      <CurText text={experimentState.curText} />
      <SuggestionsBar />
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

export const PracticeComputer = inject('state', 'dispatch')(observer(({state, dispatch}) => {
    // <h1>Practice with Phrase Suggestions</h1>
    // <TutorialTodo done={state.tutorialTasks.tasks.quadTap}>Just for fun, try a <b>quadruple-tap</b> to insert 4 words.</TutorialTodo>
    return <div>
      <p>For technical reasons, we have to use a special keyboard for this experiment. It will probably feel harder to type with than your ordinary keyboard, and it's missing some characters you may want to type, sorry about that.
      But it has a few special features that we want to show you!</p>

      <p>In this <b>practice round</b>, we'll pretend to write the opening sentences to a US <b>State of the Union address</b> in an imaginary world where a <b>united Africa has become a superpower</b>.</p>

      <p>Let's start off with a standard opening: "members of congress, my fellow americans" (we're going to not worry about capitalization). Notice the 3 boxes above the keyboard. Each one shows a word that you can insert by tapping on the box.</p>
      <TutorialTodo done={state.tutorialTasks.tasks.tapSuggestion}>Try a single <b>tap</b> on the "members" box to insert that word.</TutorialTodo>

      <p>Sometimes the boxes above the keyboard will show a complete phrase, starting with the highlighted word.
      Tap a box to insert words from that phrase, one word per tap. So if you want the first two words, double-tap; if you want the first 4 words, tap 4 times.</p>
      <TutorialTodo done={state.tutorialTasks.tasks.doubleTap}>Now try a <b>double-tap</b> to insert two words.</TutorialTodo>

      <p>Inevitably you'll want to write something different from the words or phrases provided. In that case, just tap the letters on the keyboard.</p>

      {['typeKeyboard', 'backspace', 'specialChars'].map(name => <TutorialTodo key={name} done={state.tutorialTasks.tasks[name]}>{tutorialTaskDescs[name]}</TutorialTodo>)}


      {state.experimentState.useConstraints.letter && <p>For fun (or at least for a challenge), <b>certain letters will be unusable</b>. The letter will change each sentence. The phrases suggestions obey the constraint, so they may help you.</p>}
      <p>Occasionally, double-tapping may cause your phone to zoom its screen. Unfortunately there's not much we can do about that. If that happens, try double-tapping on an empty area, or reload the page (you won't lose your work).</p>
      <p>Don't worry about capitalization, numbers, or anything else that isn't on the keyboard.</p>
      {_.every(['typeKeyboard', 'backspace', 'specialChars', 'tapSuggestion', 'doubleTap'].map(name => state.tutorialTasks.tasks[name])) ? <p>
        Now that you know how it works, <b>try writing another sentence, just for practice (have fun with it!). Use both the keys and the suggestions.</b><br/>
        When you're ready to move on, click here: <NextBtn />.</p> : <p>Complete all of the tutorial steps to move on.</p>}
    </div>;
  }));

export const PracticeComputer2 = () => <div>
    <h1>Practice with Session B Phrase Suggestions</h1>

    <p>We are now starting the second of two writing sessions, Session B. In this session, <b>the suggestions will show different kinds of phrases</b>. Other than that, nothing changed.</p>

    <p>We put up the practice keyboard on your phone again so you can try out the different phrase suggestions. <b>Try writing a few sentences</b> to get a feel for the new suggestions. Use both the keys and the suggestions.</p>
    <p>When you're comfortable with the new phrases, click this button to move on: <NextBtn /></p>
  </div>;

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
      <li>Scan this:<br/><img src={"https://zxing.org/w/chart?cht=qr&chs=350x350&chld=L&choe=UTF-8&chl=" + encodeURIComponent(url)} role="presentation"/></li>
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
