import Promise from "bluebird";
import { MasterStateStore } from "./MasterStateStore";
import { readLogFile } from './testUtil.js';
import { processLog } from './Analyzer.js';

const participantIds = ["99c66d"];
let logData = {};

beforeAll(() => {
  return Promise.map(participantIds, readLogFile)
    .then(logs => {
      console.log(`Loaded ${logs.length} logs.`);
      logData = logs;
    })
    .catch(err => console.error(err));
});

it("processes logs without crashing", () => {
  logData.forEach(([participantId, log]) => {
    processLog(log);
  });
});
