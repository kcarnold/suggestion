import pako from 'pako';

export default function WSClient(path) {
  var self = this;
  self.attempts = 0;
  self.waitingToReconnect = false;
  self.helloMsgs = [];

  self.connect = () => {
    self.attempts++;
    self.waitingToReconnect = false;
    self.ws = new WebSocket(path);
    self.ws.onopen = onopen;
    self.ws.onclose = onclose;
    self.ws.onerror = onerror;
    self.ws.onmessage = onmessage;
  };

  self.queue = [];

  function _send(msg) {
    self.deflater.push(JSON.stringify(msg), pako.Z_SYNC_FLUSH);
    self.ws.send(self.deflater.result);
  }

  setInterval(function() {
    if (self.ws.readyState === WebSocket.OPEN) {
      self.send({type: 'ping'});
    }
  }, 10000);

  function onopen() {
    self.attempts = 0;
    console.log('ws open', self.ws.readyState, self.queue);
    self.deflater = new pako.Deflate({to: 'string'});
    self.inflater = new pako.Inflate({to: 'string'});
    for (var i=0; i<self.helloMsgs.length; i++) {
        _send(self.helloMsgs[i]);
    }
    while(self.queue.length) {
      _send(self.queue.shift());
    }
    self.stateChanged && self.stateChanged('open');
  }

  function backoffReconnect() {
    if (self.waitingToReconnect) return;
    self.waitingToReconnect = true;
    setTimeout(self.connect, Math.min(10, Math.pow(2, self.attempts - 2)) * 1000);
  }

  function onclose() {
    console.log('websocket closed');
    self.stateChanged && self.stateChanged('closed');
    backoffReconnect();
  }

  function onerror(e) {
    console.log('websocket error', e);
    backoffReconnect();
  }

  function onmessage(message) {
    self.inflater.push(message.data, pako.Z_SYNC_FLUSH);
    var data = JSON.parse(self.inflater.result);
    self.onmessage(data);
  }

  self.send = function(msg) {
    msg = {timestamp: +new Date(), ...msg};
    if (self.ws.readyState === WebSocket.OPEN) {
        _send(msg);
    } else {
        self.queue.push(msg);
    }
    return msg;
  }

  self.isOpen = function() {
    return self.ws.readyState === WebSocket.OPEN;
  };

  self.setHello = function(msgs) {
    self.helloMsgs = msgs;
  };

  return self;
}
