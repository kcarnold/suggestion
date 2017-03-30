import React, { Component } from 'react';
import M from 'mobx';
import {observer} from 'mobx-react';

import Consent from './Consent';

export class State {
  constructor(text) {
    M.extendObservable(this, {
      responses: new M.ObservableMap()
    });
  }

  getPage(page) {
    return this.responses.get(page) || new M.ObservableMap();
  }

  setResponse = M.action((page, item, val) => {
    if (!this.responses.has(page)) {
      this.responses.set(page, new M.ObservableMap())
    }
    this.responses.get(page).set(item, val);
  });
}


let state = new State();
window.state = state;

export const RatingPage = observer(class RatingPage extends Component {
  render() {
    let {pageNum, page, state} = this.props;
    let {context, options} = page;
    context = context.slice(-50);
    let scaleItems = ["extremely unnatural", 'somewhat unnatural', 'somewhat natural', 'extremely natural'];
    let ratingPage = state.getPage(pageNum);

    return <div className="RatingPage">
      <h3>Page {pageNum+1}</h3>

      <table className="ratings-table">
        <thead><tr><th></th>{scaleItems.map((item, i) =><th key={i}>{item}</th>)}</tr></thead>
        <tbody>
          {options.map(([cond, phrase], phraseIdx) => <tr key={phraseIdx}>
            <td>...{context}<br/><b>{phrase}</b> ...</td>
            {scaleItems.map((x, scaleIdx) => <td key={scaleIdx}>
              <input type="radio"
                checked={ratingPage.get(phraseIdx) === scaleIdx}
                onChange={() => {console.log(pageNum, phraseIdx, scaleIdx); state.setResponse(pageNum, phraseIdx, scaleIdx);}} />
            </td>)}</tr>)}
        </tbody>
      </table>
    </div>;
  }
});

export const App = observer(class App extends Component {
  state = {consented: true};

  render() {
    let {data} = this.props;
    let pages = data;
    let {consented} = this.state;

    if (!consented) {
      return <Consent onConsented={() => {
          this.setState({consented: true});
          setTimeout(() => {window.scrollTo(0, 0);}, 100);
        }} />;
    }


    return <div className="App">
      <div style={{background: 'yellow', boxShadow: '1px 1px 4px grey'}}>
      <h1>Instructions</h1>
      <p>How natural is the bolded part of each sentence?</p>
      <p>The sentence will be cut off after the bolded part. If the sentence is not over at that point, that's fine, imagine that the rest of the sentence is natural.</p>
      </div>

      {pages.map((page, pageNum) => <RatingPage key={pageNum} pageNum={pageNum} page={page} state={state} />)}

      <br/><br/>

      <input type="text" readOnly={true} name="results" value={JSON.stringify({state})} />
    </div>;
  }
});

export default App;
