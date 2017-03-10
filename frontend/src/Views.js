import React, { Component } from 'react';
import _ from 'lodash';
import {observer, inject} from 'mobx-react';
import StarRatingComponent from 'react-star-rating-component';
import {Keyboard} from './Keyboard';

const surveyURLs = {
  intro: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_9GiIgGOn3Snoxwh',
  postFreewrite: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_0OCqAQl6o7BiidT',
  postTask: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_5yztOdf3SX8EtOl',
  postExp: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_8HVnUso1f0DZExv',
}

class Suggestion extends Component {
  render() {
    let {onTap, word, preview, isValid} = this.props;
    return <div
      className={"Suggestion" + (isValid ? '' : ' invalid')}
      onTouchStart={isValid ? onTap : null}>
      {word}<span className="preview">{preview.join(' ')}</span>
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


const RedirectToSurvey = inject('clientId', 'clientKind', 'spying')(class RedirectToSurvey extends Component {
  componentDidMount() {
    if (this.props.spying) return;
    // This timeout is necessary to give the current page enough time to log the event that caused this render.
    // 2 seconds is probably overdoing it, but on the safe side.
    this.timeout = setTimeout(() => {
      let afterEvent =  this.props.afterEvent || 'completeSurvey';
      let nextURL = `${window.location.protocol}//${window.location.host}/?${this.props.clientId}-${this.props.clientKind}#${afterEvent}`;
      window.location.href = `${this.props.url}&clientId=${this.props.clientId}&nextURL=${encodeURIComponent(nextURL)}`;
    }, 2000);
  }

  componentWillUnmount() {
    // Just in case.
    clearTimeout(this.timeout);
  }

  render() {
    return <div>redirecting...</div>;
  }
});

const TutorialTodo = ({done, children}) => <div style={{color: done ? 'green' : 'red'}}>{done ? '\u2611' : '\u2610'} {children}</div>;

class CurText extends Component {
  componentDidMount() {
    this.cursor.scrollIntoView();
  }

  componentDidUpdate() {
    this.cursor.scrollIntoView();
  }

  render() {
    return <div className="CurText">{this.props.text}<span className="Cursor" ref={elt => {this.cursor = elt;}}></span></div>;
  }
}

export const screenViews = {
  Welcome: () => <div>
    <h1>Welcome</h1>
    <p>By continuing, you agree that you have been provided with the consent form
    for this study and agree to its terms.</p>
    <NextBtn /></div>,

  ProbablyWrongCode: () => <div>
    <p>Waiting for computer. If you're seeing this on your phone, you probably mistyped your code.</p>
  </div>,

  SelectRestaurants: inject('state')(observer(({state}) => <div>
    <p>Great! Now let's get ready for the experiment.</p>
    <p>Think of 2 restaurants or cafes you've been to recently.</p>
    <div>1. <ControlledInput name="restaurant1"/><br />When were you last there? <ControlledInput name="visit1"/>
      <br />How would you rate that visit? <ControlledStarRating name="star1" />
    </div>
    <div>2. <ControlledInput name="restaurant2"/><br /> When were you last there? <ControlledInput name="visit2"/>
      <br />How would you rate that visit? <ControlledStarRating name="star2" />
    </div>
    <p>(The Next button will be enabled once all fields are filled out.)</p>
    <NextBtn disabled={!_.every('restaurant1 visit1 star1 restaurant2 visit2 star2'.split(' '), x => state.controlledInputs.get(x))} />
  </div>)),

  Instructions: inject('state')(observer(({state}) => <div>
    <h1>Ready to write a review?</h1>
    <p>Think about your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b>.</p>
    <p>Let's write a review of this experience, like you might see on a site like Yelp or TripAdvisor. We'll do this in <b>two steps</b>:</p>
    <ol>
      <li style={{paddingBottom: '1em'}}><b>Explore what you might want to talk about</b> by typing whatever comes to mind. Don't worry about grammar, coherence, accuracy, etc. ({state.times.prewriteTimer / 60} minutes)</li>
      <li>Type out the <b>most detailed review you can</b>. ({state.times.finalTimer / 60} minutes)</li>
    </ol>
    <p>Click Next when you're ready to start Step 1. You will have {state.nextScreen.timer / 60} minutes (note the timer on top). (If you need a break, this would be a good time.)</p>
    <NextBtn /></div>)),

  RevisionComputer: inject('state')(observer(({state}) => <div>
      <h1>Revise</h1>
      <p>Here is what you wrote last time:</p>
      <div style={{whiteSpace: 'pre-line'}}>{state.prevExperimentState.curText}</div>
    </div>)),

  ExperimentScreen: inject('state', 'dispatch')(observer(({state, dispatch}) => {
      let {experimentState, curScreen} = state;
      return <div className="ExperimentScreen">
        <div className="header">
          {curScreen.isPrewrite ? "Rough draft" : "Revised"} review for your <b>{state.curPlace.visit}</b> visit to <b>{state.curPlace.name}</b> ({state.curPlace.stars} stars)
          <div style={{float: 'right'}}><Timer /></div>
        </div>
        <CurText text={experimentState.curText} />
        <SuggestionsBar />
        <Keyboard dispatch={dispatch} />
      </div>;
    })),

  PrewriteInstructionsDuring: () => <div>
    <h1>Brainstorm what you might want to say</h1>

    <p>Use your phone to type anything that comes to mind. Don't worry about grammar, coherence, accuracy, or anything else.</p>

    <p>When time is up, the experiment will automatically advance.</p>
  </div>,


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
    let {experimentState} = state;
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

  BreakBeforeEdit: inject('state')(observer(({state}) => <div>
    <p>Now, try to write the <b>most detailed review you can</b>. You'll be using the same keyboard as you just used.
    You'll have {state.nextScreen.timer / 60} minutes.</p>
    <NextBtn />
    </div>)),

  BreakBeforeEditPhone: () => <div>Time is up. Follow the instructions on your computer.</div>,

  EditScreen: inject('state', 'dispatch')(observer(({state, dispatch}) => <div className="EditPage">
    <div style={{backgroundColor: '#ccc', color: 'black'}}>
      Now, edit what you wrote to make it better. <Timer />
    </div>
    <textarea value={state.curEditText} onChange={evt => {dispatch({type: 'controlledInputChanged', name: state.curEditTextName, value: evt.target.value});}} />;
  </div>)),

  IntroSurvey: () => <RedirectToSurvey url={surveyURLs.intro} />,
  PostFreewriteSurvey: () => <RedirectToSurvey url={surveyURLs.postFreewrite} />,
  PostTaskSurvey: () => <RedirectToSurvey url={surveyURLs.postTask} />,
  PostExpSurvey: () => <RedirectToSurvey url={surveyURLs.postExp} />,
  Done: inject('clientId')(({clientId}) => <div>Thanks! Your code is {clientId}.</div>),
  LookAtPhone: inject('clientId')(({clientId}) => <div><p>Complete this step on your phone.</p> If you need it, your phone code is <tt>{clientId}-p</tt>.</div>),
  LookAtComputer: inject('clientId')(({clientId}) => <div><p>Complete this step on your computer.</p> If you need it, your computer code is <tt>{clientId}-c</tt>.</div>),
  SetupPairingComputer: inject('clientId')(({clientId}) => <div>
    <p>You will need two devices to complete this study: a <b>laptop/desktop computer</b> (you could use a tablet but we haven't tested it), and a <b>smartphone</b> with a web browser and WiFi (we will not be responsible for any data charges).</p>

    <p>On your phone's web browser, go to <tt>megacomplete.net</tt> and enter <tt>{clientId}-p</tt>. There will be a button on that page to continue.</p>
    <p>If you have a barcode reader on your phone, you can use scan this:<br/><img src={"https://zxing.org/w/chart?cht=qr&chs=350x350&chld=L&choe=UTF-8&chl=" + encodeURIComponent("http://megacomplete.net/?" + clientId + "-p")} role="presentation"/></p>
  </div>),
  SetupPairingPhone: () => <div>Successfully paired! <NextBtn /></div>,
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
      {React.createElement(screenViews[screenName])}
    </div>);
}));
