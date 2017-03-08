import React, { Component } from 'react';
import _ from 'lodash';
import {observer, inject} from 'mobx-react';
import StarRatingComponent from 'react-star-rating-component';
import {Keyboard} from './Keyboard';

const surveyURLs = {
  // postTask: 'https://harvard.az1.qualtrics.com/SE/?SID=SV_8FWK07Bfg4Xv2br',
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

function approxTime(remain) {
  if (remain > 60)
    return '~' + Math.round(remain / 60) + ' min';
  if (remain > 10)
    return '~' + (Math.round(remain / 10) * 10) + ' sec';
  return Math.ceil(remain) + ' sec';
}

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
    return <div className="timer">{approxTime(remain)}</div>;
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
      let nextURL = `${window.location.protocol}//${window.location.host}/?${this.props.clientId}-${this.props.clientKind}#${this.props.afterEvent}`;
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
    <p>Great, the phone is paired. Let's get ready for the experiment.</p>
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
    <p>Let's write a review of this experience (like you might see on a site like Yelp or Google Maps). We'll do this in <b>two steps</b>:</p>
    <ol>
      <li style={{paddingBottom: '1em'}}><b>Explore what you might want to talk about</b> by typing whatever comes to mind. Don't worry about grammar, coherence, accuracy, etc. ({state.times.prewriteTimer / 60} minutes)</li>
      <li>Type out the <b>most detailed review you can</b>. ({state.times.finalTimer / 60} minutes)</li>
    </ol>
    <p>Tap Next when you're ready to start Step 1. You will have {state.nextScreen.timer / 60} minutes (note the timer on top). (If you need a break, this would be a good time.)</p>
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
          <div style={{float: 'right'}}><Timer /> ({state.block === 0 ? 'A' : 'B'})</div>
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
    let {experimentState} = state;
    let suggs = experimentState.visibleSuggestions;
    if (experimentState.lastSuggestionsFromServer.length === 0 && experimentState.contextSequenceNum === 0) {
      return <div>Loading...</div>;
    }
    return <div>
      <h1>Practice with Phrase Suggestions (version A)</h1>
      <p>This experiment uses a special mobile phone keyboard that gives <i>phrase</i> suggestions. Let's practice using them.</p>
      <p>Notice the 3 boxes above the keyboard. Each one shows a phrase, with the left-most word highlighted. Tapping a box inserts the highlighted word and moves on to the next word in the phrase.</p>
      <TutorialTodo done={state.tutorialTasks.tasks.tapSuggestion}><b>Tap</b> the leftmost box  to insert &ldquo;<tt>{suggs && suggs[0].words[0]}</tt>&rdquo;.</TutorialTodo>
      <TutorialTodo done={state.tutorialTasks.tasks.doubleTap}>Now <b>double-tap</b> the middle box to insert &ldquo;<tt>{suggs && suggs[1].words.slice(0,2).join(' ')}</tt>&rdquo;. </TutorialTodo>
      <TutorialTodo done={state.tutorialTasks.tasks.tripleTap}>Now <b>triple-tap</b> the rightmost box to insert &ldquo;<tt>{suggs && suggs[2].words.slice(0,3).join(' ')}</tt>&rdquo;. </TutorialTodo>
      <TutorialTodo done={state.tutorialTasks.tasks.typeKeyboard}>Now <b>type a word on the keyboard</b>.  </TutorialTodo>
      <p>Don't worry about capitalization, numbers, or anything else that isn't on the keyboard.</p>
      {state.tutorialTasks.allDone && <p>
        Now that you know how it works, <b>try writing a few sentences to get some more practice. Use both the keys and the suggestions.</b><br/>
        When you're ready to move on, tap <NextBtn />.</p>}
    </div>;
  })),

  PracticeComputer2: () => <div>
    <h1>Practice with Phrase Suggestions (version B)</h1>

    <p>There will be two versions of the phrase suggestions. You just tried version A, now try out version B.</p>

    <p><b>Try writing a few sentences to get some more practice. Use both the keys and the suggestions.</b></p>
    <p><NextBtn /></p>
  </div>,

  BreakBeforeEdit: inject('state')(observer(({state}) => <div>
    <p>Time is up. Now, try to write the <b>most detailed review you can</b>. You'll be using the same keyboard as you just used.
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

  PostTaskSurvey: () => <RedirectToSurvey url={surveyURLs.postTask} afterEvent={'completeSurvey'} />,
  PostExpSurvey: () => <RedirectToSurvey url={surveyURLs.postExp} afterEvent={'completeSurvey'} />,
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
      <div className="clientId">{state.clientId}</div>
    </div>);
}));
