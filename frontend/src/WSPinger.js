import _ from 'lodash';

export function doPing(url, times, callback) {
  let rtts = [];
  let lastSendTime = null;
  let ws = new WebSocket(url);
  const sendPing = () => {
    lastSendTime = +new Date();
    ws.send(`ping`);
  };

  ws.onopen = () => {
    setTimeout(sendPing, 50);
  };

  ws.onmessage = (msg) => {
    rtts.push(+new Date() - lastSendTime);
    if (rtts.length === times) {
      callback({rtts, mean: _.mean(rtts)});
    } else {
      sendPing();
    }
  };
}
