// @flow
import * as M from 'mobx';
import _ from 'lodash';

let multiTapThresholdMs = 500;

export default class TutorialTasks {
  tasks: {[name: string]: boolean} ;
  consectutiveTaps: Object;

  constructor() {
    M.extendObservable(this, {
      consectutiveTaps: {},
      tasks: {
        tapSuggestion: false,
        doubleTap: false,
        quadTap: false,
        typeKeyboard: false,
        backspace: false,
        megaBackspace: false,
        undo: false,
        specialChars: false,
        tapPrediction: false,
        tapAlternative: false,
      },
      get allDone() {
        let {tasks} = this;
        return _.every(tasks);
      }
    });
  }

  handleEvent(event) {
    let timestamp = event.jsTimestamp;
    switch(event.type) {
    case 'tapSuggestion':
      this.tasks['tapSuggestion'] = true;
      if (event.which === 'predictions') {
        this.tasks['tapPrediction'] = true;
      } else if (event.which === 'synonyms') {
        this.tasks['tapAlternative'] = true;
      }
      if (this.consectutiveTaps.slot === event.slot && timestamp - this.consectutiveTaps.lastTimestamp < multiTapThresholdMs) {
        this.consectutiveTaps.times++;
        this.consectutiveTaps.lastTimestamp = timestamp;
        if (this.consectutiveTaps.times >= 2) {
          this.tasks.doubleTap = true;
        }
        if (this.consectutiveTaps.times >= 4) {
          this.tasks.quadTap = true;
        }
      } else {
        this.consectutiveTaps = {slot: event.slot, times: 1, lastTimestamp: event.jsTimestamp};
      }
      break;
    case 'tapKey':
      if (event.key.match(/[a-z]/)) {
        this.tasks.typeKeyboard = true;
      } else if (event.key.match(/[-.,!'?]/)) {
        this.tasks.specialChars = true;
      }
      this.consectutiveTaps = {};
      break;
    case 'tapBackspace':
      this.consectutiveTaps = {};
      this.tasks.backspace = true;
      if (event.delta < -5) {
        this.tasks.megaBackspace = true;
      }
      break;
    case 'undo':
      this.consectutiveTaps = {};
      this.tasks.undo = true;
      break;
    default:
    }
  }
}
