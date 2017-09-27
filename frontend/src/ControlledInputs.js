import React from 'react';
import StarRatingComponent from 'react-star-rating-component';
import {observer, inject} from 'mobx-react';


export const ControlledInput = inject('dispatch', 'state', 'spying')(observer(function ControlledInput({state, dispatch, name, multiline, spying, ...props}) {
  let proto = multiline ? 'textarea' : 'input';
  let innerProps = {};
  if (!multiline) {
    innerProps.autoComplete = "off";
    innerProps.autoCorrect="off";
    innerProps.autoCapitalize="on";
    innerProps.spellCheck="false";
  }
  innerProps.name = name;
  innerProps.onChange = evt => {dispatch({type: 'controlledInputChanged', name, value: evt.target.value});}
  innerProps.value=state.controlledInputs.get(name) || '';
  if (spying) {
    innerProps['title'] = name;
  }
  return React.createElement(proto, {...innerProps, ...props});
}));


export const ControlledStarRating = inject('dispatch', 'state')(observer(({state, dispatch, name}) => <div>
  <StarRatingComponent
    name={name} starCount={5} value={state.controlledInputs.get(name) || 0}
    onStarClick={value => {dispatch({type: 'controlledInputChanged', name, value});}}
    renderStarIcon={(idx, value) => <i style={{fontStyle: 'normal'}}>{idx<=value ? '\u2605' : '\u2606'}</i>} />
  {state.controlledInputs.get(name)}
  </div>));


