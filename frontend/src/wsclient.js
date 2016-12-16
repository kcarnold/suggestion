export default function WSClient(path) {
  var self = this;
  self.attempts = 0;
  self.waitingToReconnect = false;
  self.helloMsgs = [];

  function tryConnect() {
    self.attempts++;
    self.waitingToReconnect = false;
    self.ws = new WebSocket(path);
    self.ws.onopen = onopen;
    self.ws.onclose = onclose;
    self.ws.onerror = onerror;
    self.ws.onmessage = onmessage;
  }

  self.queue = [];

  setInterval(function() {
    if (self.ws.readyState === WebSocket.OPEN) {
      self.send({type: 'ping'});
    }
  }, 10000);

  function onopen() {
    self.attempts = 0;
    console.log('ws open', self.ws.readyState, self.queue);
    self.stateChanged && self.stateChanged('open');
    for (var i=0; i<self.helloMsgs.length; i++) {
        self.ws.send(JSON.stringify(self.helloMsgs[i]));
    }
    while(self.queue.length) {
      self.ws.send(JSON.stringify(self.queue.shift()));
    }
  }

  function backoffReconnect() {
    if (self.waitingToReconnect) return;
    self.waitingToReconnect = true;
    setTimeout(tryConnect, Math.min(10, Math.pow(2, self.attempts - 2)) * 1000);
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
    var data = JSON.parse(message.data);
    self.onmessage(data);
  }

  self.send = function(msg) {
    msg = {timestamp: +new Date(), ...msg};
    if (self.ws.readyState === WebSocket.OPEN) {
        self.ws.send(JSON.stringify(msg));
    } else {
        self.queue.push(msg);
    }
  }

  self.sendHello = function(msg) {
    self.helloMsgs.push(msg);
    if (self.ws.readyState === WebSocket.OPEN) {
        self.ws.send(JSON.stringify(msg));
    }
  }

  tryConnect();
  return self;
}
