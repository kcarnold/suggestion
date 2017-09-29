import React, { Component } from 'react';
import _ from 'lodash';

var KEYLABELS = {
    ' ': 'space',
    '⌫': '',
    '\r': 'undo',
};

function getClosestKey(keyRects, touchX, touchY) {
    var closestKey = null, closestDist = Infinity;
    keyRects.forEach(function(krect) {
        var rect = krect.rect, hwidth = rect.width / 2, hheight = rect.height / 2, x = rect.left + hwidth, y = rect.top + hheight;
        var dx = Math.max(0, Math.abs(touchX - x) - hwidth), dy = Math.max(0, Math.abs(touchY - y) - hheight);
        var dist = dx * dx + dy * dy;
        if (dist < closestDist) {
            closestDist = dist;
            closestKey = krect.key;
        }
    });
    return closestKey;
}

export class Keyboard extends Component {
  lastKbdRect = null;
  deleteZeroX = null;
  lastUpdateDelta = null;

  handleTouchStart = (evt) => {
    let {dispatch} = this.props;
    let {clientX, clientY} = evt.type === 'touchstart' ? evt.changedTouches[0] : evt;
    let {top, left, width, height} = this.node.getBoundingClientRect();
    let kbdRect = {top, left, width, height};
    if (!_.isEqual(kbdRect, this.lastKbdRect)) {
      this.lastKbdRect = kbdRect;
      var keyRects = [];
      this.keyRects = keyRects;
      _.forOwn(this.keyNodes, (node, key) => {
        let {top, left, width, height} = node.getBoundingClientRect();
        this.keyRects.push({rect: {top, left, width, height}, key});
      });
    }

    let key = getClosestKey(this.keyRects, clientX, clientY);
    if (key === '\r') {
      dispatch({type: 'undo'});
    } else if (key === '⌫') {
      this.deleteZeroX = clientX + 5;
      this.lastUpdateDelta = -1;
      dispatch({type: 'updateDeleting', delta: -1});
    } else {
      dispatch({type: 'tapKey', key, x: clientX, y: clientY});
    }
    evt.preventDefault();
    evt.stopPropagation();
  };

  handleTouchMove = (evt) => {
    let {deleteZeroX, lastUpdateDelta} = this;
    if (deleteZeroX) {
      let delta = Math.round((evt.targetTouches[0].clientX - deleteZeroX) / 5);
      if (delta !== lastUpdateDelta) {
        this.props.dispatch({type: 'updateDeleting', delta: delta});
        this.lastUpdateDelta = delta;
      }
    }
    evt.preventDefault();
    evt.stopPropagation();
  }

  handleTouchEnd = (evt) => {
    if (this.deleteZeroX) {
      this.props.dispatch({type: 'tapBackspace', delta: this.lastUpdateDelta});
      this.deleteZeroX = null;
      this.lastUpdateDelta = null;
    }
    evt.preventDefault();
    evt.stopPropagation();
  };

  render() {
    var keyNodes = {};
    this.keyNodes = keyNodes;
    return <div className="Keyboard" ref={node => this.node = node}
      onTouchStart={this.handleTouchStart}
      onTouchMove={this.handleTouchMove}
      onTouchEnd={this.handleTouchEnd}
    >{
      ['qwertyuiop', 'asdfghjkl', '\'?zxcvbnm⌫', '-!, .\r'].map(function(row, i) {
          return <div key={i} className="row">{
            _.map(row, function(key, j) {
              // if (layer === 'upper') key = key.toUpperCase();
              var label = key in KEYLABELS ? KEYLABELS[key] : key;
              var className = 'key';
              if ('\r⌫\'-!,.?'.indexOf(key) !== -1) className += ' key-reverse';
              return <div key={key} className={className} data-key={key} ref={node => keyNodes[key] = node}>{label}</div>;
          })}</div>
          })}
      </div>;
  }
}
