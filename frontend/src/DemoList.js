import React from 'react';
import _ from 'lodash';
import {namedConditions} from './MasterStateStore';

const DemoList = () => <ul>{
    _.map(namedConditions, (val, key) => <li key={key}><a href={`?demo${key}-p`}>{key}</a></li>)
    }</ul>;
export default DemoList;
