import React from 'react';
import { mount } from 'enzyme';
import { MasterStateStore} from './MasterStateStore';
import {Provider} from 'mobx-react';
import {SelectRestaurants} from './Views';

function mockStateAndDispatch(participantId, masterConfig) {
  let store = new MasterStateStore(participantId);
  store.handleEvent({type: 'externalAction', externalAction: `c=${masterConfig}`});
  function dispatch(event) {
    store.handleEvent(event);
  }
  return { store, dispatch };
}

describe('<SelectRestaurants>', () => {
  it('gives instructions', () => {
    let {store, dispatch} = mockStateAndDispatch('zzzzzz', 'sent4');
    let component = <Provider state={store} dispatch={dispatch} spying={false}><SelectRestaurants /></Provider>;
    let wrapper = mount(component);
    expect(wrapper).toIncludeText('restaurants');
    let places = wrapper.find('.Restaurant');
    expect(places).toHaveLength(4);
    places.forEach((place, idx) => {
      idx = idx + 1;
      place.find({name: `restaurant${idx}`}).simulate('change', {target: {value: `restaurant${idx}`}});
    })
    expect(store.places[0].name.slice(0,3)).toEqual('res')
  });
});
