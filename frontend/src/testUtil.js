import Promise from "bluebird";
const readFile = Promise.promisify(require("fs").readFile);

export function readLogFile(participantId) {
  let filename = `${participantId}.jsonl`;
  if (!(participantId.slice(0, 5) === "smoke")) filename = "../logs/" + filename;
  return readFile(filename, "utf8").then(data => [
    participantId,
    data
      .split("\n")
      .filter(line => line.length > 0)
      .map(line => JSON.parse(line)),
  ]);
}
