import React from 'react';
import { mount } from 'enzyme';
import { MasterStateStore} from './MasterStateStore';
import {Provider} from 'mobx-react';
import {SelectRestaurants} from './Views';

describe('<SelectRestaurants>', () => {
  it('gives instructions', () => {
    let store = new MasterStateStore('zzzzzz');
    store.handleEvent({type: 'externalAction', externalAction: 'c=sent4'});
    let component = <Provider state={store} dispatch={() => {}}><SelectRestaurants /></Provider>;
    let wrapper = mount(component);
    expect(wrapper).toIncludeText('restaurants')
  });
});
