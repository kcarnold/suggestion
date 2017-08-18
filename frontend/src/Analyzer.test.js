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
    let result = processLog(log);
    expect(result.byExpPage).toBeDefined();
    let page = result.byExpPage['final-0'];
    expect(page.displayedSuggs.length).toBeGreaterThan(0);
    let suggEntry = page.displayedSuggs[0];
    expect(suggEntry).toMatchObject({
      timestamp: expect.any(Number),
      context: expect.any(String),
      recs: expect.anything(),
      latency: expect.any(Number),
      action: expect.objectContaining({type: expect.any(String)}),
    });
  });
});
