import React, { Component } from 'react';
import _ from 'lodash';
import {observer, inject} from 'mobx-react';
import StarRatingComponent from 'react-star-rating-component';
import {Keyboard} from './Keyboard';

const surveyURLs = {
  intro: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_9GiIgGOn3Snoxwh',
  instructionsQuiz: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_42ziiSrsZzOdBul',
  postFreewrite: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_0OCqAQl6o7BiidT',
  postTask: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_5yztOdf3SX8EtOl',
  postExp: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_8HVnUso1f0DZExv',
}

const texts = {
  overallInstructions: <span>Write the true story of your experience. Tell your reader <b>as many vivid details as you can</b>. Don’t worry about <em>summarizing</em> or <em>giving recommendations</em>.</span>,
  brainstormingInstructions: <span><b>Brainstorm what you might want to talk about</b> by typing anything that comes to mind, even if it's not entirely accurate. Don't worry about grammar, coherence, accuracy, or anything else, this is just for you.</span>,
  revisionInstructions: <span>Type out the <b>most detailed true story you can</b> about your experience.</span>,
};

class Suggestion extends Component {
  render() {
    let {onTap, word, preview, isValid} = this.props;
    return <div
      className={"Suggestion" + (isValid ? '' : ' invalid')}
      onTouchStart={isValid ? onTap : null}>
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
        isValid={sugg.isValid} />
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

const Timer = inject('dispatch', 'state')(observer(class Timer extends Component {
  state = {remain: Infinity};
  tick = () => {
    let {dispatch, state} = this.props;
    if (!state.timerStartedAt) return;
    let elapsed = (+new Date() - state.timerStartedAt) / 1000;
    let remain = state.timerDur - elapsed;
    this.setState({remain});
    if (remain > 0) {
      this.timeout = setTimeout(this.tick, 100);
    } else {
      this.timeout = null;
      advance(state, dispatch);
    }
  };

  componentDidMount() {
    this.tick();
  }

  componentWillUnmount() {
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
  }

  render() {
    let {remain} = this.state;
    if (Math.abs(remain) > 1e10) remain = 0;
    let remainMin = Math.floor(remain / 60);
    let remainSec = ('00'+Math.floor((remain - 60 * remainMin))).slice(-2);
    return <div className="timer">{remainMin}:{remainSec}</div>;
  }
}));

const ControlledInput = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <input
  onChange={evt => {dispatch({type: 'controlledInputChanged', name, value: evt.target.value});}} value={state.controlledInputs.get(name) || ''} />));

const ControlledStarRating = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <StarRatingComponent
  name={name} starCount={5} value={state.controlledInputs.get(name) || 0}
  onStarClick={value => {dispatch({type: 'controlledInputChanged', name, value});}}
  renderStarIcon={(idx, value) => <i style={{fontStyle: 'normal'}}>{idx<=value ? '\u2605' : '\u2606'}</i>} />));


const RedirectToSurvey = inject('state', 'clientId', 'clientKind', 'spying')(class RedirectToSurvey extends Component {
  componentDidMount() {
    if (this.props.spying) return;
    // This timeout is necessary to give the current page enough time to log the event that caused this render.
    // 2 seconds is probably overdoing it, but on the safe side.
    this.timeout = setTimeout(() => {
      let afterEvent =  this.props.afterEvent || 'completeSurvey';
      let nextURL = `${window.location.protocol}//${window.location.host}/?${this.props.clientId}-${this.props.clientKind}#${afterEvent}`;
      let url = `${this.props.url}&clientId=${this.props.clientId}&nextURL=${encodeURIComponent(nextURL)}`
      if (this.props.extraParams) {
        url += '&' + _.map(this.props.extraParams, (v, k) => `${k}=${v}`).join('&');
      }
      window.location.href = url;
    }, 2000);
  }

  componentWillUnmount() {
    // Just in case.
    clearTimeout(this.timeout);
  }

  render() {
    if (this.props.spying) {
      let url = this.props.url;
      if (this.props.extraParams) {
        url += '&' + _.map(this.props.extraParams, (v, k) => `${k}=${v}`).join('&');
      }
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
    return <div className="CurText">{this.props.text}<span className="Cursor" ref={elt => {this.cursor = elt;}}></span></div>;
  }
}));

export const screenViews = {
  Welcome: () => <div>
    <h1>Welcome</h1>
    <p>By continuing, you agree that you have been provided with the consent form for this study and agree to its terms.</p>
    <NextBtn /></div>,

  ProbablyWrongCode: () => <div>
    <p>Waiting for computer. If you're seeing this on your phone, you probably mistyped your code.</p>
  </div>,

  SelectRestaurants: inject('state')(observer(({state}) => <div>
    <p>Think of 2 <b>restaurants (or bars, cafes, diners, etc.)</b> you've been to recently that you <b>haven't written a review of</b>.</p>
    <div>1. <ControlledInput name="restaurant1"/><br />When were you last there? <ControlledInput name="visit1"/>
      <br />How would you rate that visit? <ControlledStarRating name="star1" />
    </div>
    <br/>
    <div>2. <ControlledInput name="restaurant2"/><br /> When were you last there? <ControlledInput name="visit2"/>
      <br />How would you rate that visit? <ControlledStarRating name="star2" />
    </div>
    <p>(The Next button will be enabled once all fields are filled out.)</p>
    <NextBtn disabled={!_.every('restaurant1 visit1 star1 restaurant2 visit2 star2'.split(' '), x => state.controlledInputs.get(x))} />
  </div>)),

  Instructions: inject('state')(observer(({state}) => {
    let inExperiment = state.curScreen.screen === 'ExperimentScreen';
    let {isPrewrite} = state;
    return <div>
      <h1>Let's write about your experience!</h1>
      <p>Think about your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>.</p>
      <p style={{border: '1px solid black', padding: '2px'}}>{texts.overallInstructions}</p>
      {state.prewrite &&  <p>We'll do this in <b>two steps</b>:</p>}
      {state.prewrite &&  <ol>
        <li style={{paddingBottom: '1em', color: state.passedQuiz && isPrewrite ? 'blue' : 'black'}}>{texts.brainstormingInstructions} ({state.times.prewriteTimer / 60} minutes)</li>
        <li style={{color: !isPrewrite ? 'blue' : 'black'}}>{texts.revisionInstructions} ({state.times.finalTimer / 60} minutes)</li>
      </ol>}
      {state.prewrite
        ? <p>Both steps will happen on your phone, using the keyboard you just practiced with.</p>
        : <p>{false && texts.revisionInstructions} You will have {state.times.finalTimer / 60} minutes.</p>}
      <hr/>
      {state.passedQuiz || inExperiment
        ? <p>Use your phone to type out {isPrewrite ? 'your brainstorming' : `your ${state.prewrite ? "revised " : ""}story`}. The experiment will automatically advance when time is up.</p>
        : <p>Your phone shows a brief quiz on these instructions. Once you've passed the quiz, look back here.</p>}
    </div>;
  })),

  ReadyPhone: inject('state')(observer(({state}) => state.passedQuiz ? <p>
    Tap Next when you're ready to start Step {state.isPrewrite ? '1' : '2'}. You will have {state.nextScreen.timer / 60} minutes (note the timer on top). (If you need a break, this would be a good time.)<br/><br/><NextBtn /></p>
    : <RedirectToSurvey url={surveyURLs.instructionsQuiz} afterEvent={'passedQuiz'} extraParams={{prewrite: state.prewrite}} />)),

/*  InstructionsQuiz: inject('state')(({state}) => state.passedQuiz ? <p>You already passed the quiz the first time, just click <NextBtn /></p> : ),*/

  RevisionComputer: inject('state')(observer(({state}) => <div>
      {texts.revisionInstructions}
      {state.prewrite && <div>
        <p>Here is what you wrote last time:</p>
        <div style={{whiteSpace: 'pre-line'}}>{state.experiments.get(`pre-${state.block}`).curText}</div>
      </div>}
    </div>)),

  ExperimentScreen: inject('state', 'dispatch')(observer(({state, dispatch}) => {
      let {experimentState} = state;
      return <div className="ExperimentScreen">
        <div className="header">
          {state.isPrewrite ? "Brainstorming for your" : "Revised"} story about your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b> ({state.curPlace.stars} stars)
          <div style={{float: 'right'}}><Timer /></div>
        </div>
        <CurText text={experimentState.curText} />
        <SuggestionsBar />
        <Keyboard dispatch={dispatch} />
      </div>;
    })),

  PracticePhone: inject('state', 'dispatch')(observer(({state, dispatch}) => {
    let {experimentState} = state;
    return <div className="ExperimentScreen">
      <div className="header">See computer for instructions. <div style={{float: 'right'}}>({state.block === 0 ? 'A' : 'B'})</div></div>
      <CurText text={experimentState.curText} />
      <SuggestionsBar />
      <Keyboard dispatch={dispatch} />
    </div>;
  })),

  PracticeComputer: inject('state', 'dispatch')(observer(({state, dispatch}) => {
    return <div>
      <p>There will be two writing sessions, Session A and Session B. We are now starting Session A.</p>
      <h1>Practice with Phrase Suggestions</h1>
      <p>This experiment uses a special mobile phone keyboard that gives <i>phrase</i> suggestions. Let's practice using them.</p>
      <p>Notice the 3 boxes above the keyboard. Each one shows a phrase. Tap a box to insert words from that phrase, one word per tap. So if you want the first two words, double-tap; if you want the first 4 words, tap 4 times.</p>
      <TutorialTodo done={state.tutorialTasks.tasks.tapSuggestion}>Try a single <b>tap</b> to insert a word.</TutorialTodo>
      <TutorialTodo done={state.tutorialTasks.tasks.doubleTap}>Now try a <b>double-tap</b> to insert two words.</TutorialTodo>
      <TutorialTodo done={state.tutorialTasks.tasks.quadTap}>Now try a <b>quadruple-tap</b> to insert 4 words.</TutorialTodo>
      <TutorialTodo done={state.tutorialTasks.tasks.typeKeyboard}>Now <b>type a word on the keyboard</b>.  </TutorialTodo>
      <p>Don't worry about capitalization, numbers, or anything else that isn't on the keyboard.</p>
      {state.tutorialTasks.allDone && <p>
        Now that you know how it works, <b>try writing a few sentences to get some more practice. Use both the keys and the suggestions.</b><br/>
        When you're ready to move on, click here: <NextBtn />.</p>}
    </div>;
  })),

  PracticeComputer2: () => <div>
    <h1>Practice with Session B Phrase Suggestions</h1>

    <p>We are now starting the second of two writing sessions, Session B. In this session, <b>the suggestions will show different kinds of phrases</b>. Other than that, nothing changed; you'll still tap once per word you want.</p>

    <p>We put up the practice keyboard on your phone again so you can try out the different phrase suggestions. <b>Try writing a few sentences to get some more practice. Use both the keys and the suggestions.</b></p>
    <p>Once you've gotten some practice, click this button to move on: <NextBtn /></p>
  </div>,

  TimesUpPhone: () => <div>Time is up. Follow the instructions on your computer.</div>,

  EditScreen: inject('state', 'dispatch')(observer(({state, dispatch}) => <div className="EditPage">
    <div style={{backgroundColor: '#ccc', color: 'black'}}>
      Now, edit what you wrote to make it better. <Timer />
    </div>
    <textarea value={state.curEditText} onChange={evt => {dispatch({type: 'controlledInputChanged', name: state.curEditTextName, value: evt.target.value});}} />;
  </div>)),

  IntroSurvey: () => <RedirectToSurvey url={surveyURLs.intro} />,
  PostFreewriteSurvey: () => <RedirectToSurvey url={surveyURLs.postFreewrite} />,
  PostTaskSurvey: inject('state')(({state}) => <RedirectToSurvey url={surveyURLs.postTask} extraParams={{prewrite: state.prewrite}} />),
  PostExpSurvey: () => <RedirectToSurvey url={surveyURLs.postExp} />,
  Done: inject('clientId')(({clientId}) => <div>Thanks! Your code is <tt>{clientId}</tt>.</div>),
  LookAtPhone: inject('clientId')(({clientId}) => <div><p>Complete this step on your phone.</p> If you need it, your phone code is <tt>{clientId}-p</tt>.</div>),
  LookAtComputer: inject('clientId')(({clientId}) => <div><p>Complete this step on your computer.</p> If you need it, your computer code is <tt>{clientId}-c</tt>.</div>),
  SetupPairingComputer: inject('clientId')(({clientId}) => <div>
    <p>You will need two devices to complete this study: a <b>laptop/desktop computer</b> (you could use a tablet but we haven't tested it), and a <b>smartphone</b> with a web browser and WiFi (we will not be responsible for any data charges).</p>

    <div>How to pair your phone (they're all the same, pick the easiest one for you):</div>
    <ul>
      <li>On your phone's web browser, go to <tt>megacomplete.net</tt> and enter <tt>{clientId}-p</tt>.</li>
      <li>Send this link to yourself: <input readOnly={true} style={{fontFamily: 'monospace', width: '25em'}} value={`http://megacomplete.net/?${clientId}-p`} /></li>
      <li>Scan this:<br/><img src={"https://zxing.org/w/chart?cht=qr&chs=350x350&chld=L&choe=UTF-8&chl=" + encodeURIComponent("http://megacomplete.net/?" + clientId + "-p")} role="presentation"/></li>
    </ul>
    <p>Once your phone is paired, there will be a button on that page to continue.</p>
  </div>),
  SetupPairingPhone: () => <div>Successfully paired! <NextBtn /></div>,

  ShowReviews: inject('state')(observer(({state}) => <div>
    <p>Here are the stories you wrote:</p>
    <h1>Session A ({state.places[0].name})</h1>
    <div style={{border: '1px solid black', margin: '5px'}}>{state.experiments.get('final-0').curText}</div>

    <h1>Session B ({state.places[1].name})</h1>
    <div style={{border: '1px solid black', margin: '5px'}}>{state.experiments.get('final-1').curText}</div>

  </div>)),
};

const shouldShowLabelOnScreen = {
  Instructions: true,
  PracticeComputer: true,
  PracticeComputer2: true,
  RevisionComputer: true,
};

export const MasterView = inject('state')(observer(({state, kind}) => {
  if (state.replaying) return <div>Loading...</div>;
  let screenDesc = state.screens[state.screenNum];
  let screenName;
  if (kind === 'c') {
    screenName = screenDesc.controllerScreen || 'LookAtPhone';
  } else {
    screenName = screenDesc.screen || 'LookAtComputer';
  }
  return (
    <div className="App">
      {kind === 'c' && shouldShowLabelOnScreen[screenName] && <div style={{float: 'right'}}>{state.blockName}</div>}
      {React.createElement(screenViews[screenName])}
    </div>);
}));
