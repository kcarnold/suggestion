import Promise from "bluebird";
const readFile = Promise.promisify(require("fs").readFile);
import { MasterStateStore } from "./MasterStateStore";

const participantIds = ["smoke0"];
let logData = {};

function readLogFile(participantId) {
  let filename = `${participantId}.jsonl`;
  if (!participantId.slice(0, 5) === "smoke") filename = "../logs/" + filename;
  return readFile(filename, "utf8").then(data => [
    participantId,
    data
      .split("\n")
      .filter(line => line.length > 0)
      .map(line => JSON.parse(line)),
  ]);
}

beforeAll(() => {
  return Promise.map(participantIds, readLogFile)
    .then(logs => {
      console.log(`Loaded ${logs.length} logs.`);
      logData = logs;
    })
    .catch(err => console.error(err));
});

it("creates state without crashing", () => {
  logData.forEach(([participantId, log]) => {
    var state = new MasterStateStore(participantId);
    log.forEach(entry => state.handleEvent(entry));
  });
});
