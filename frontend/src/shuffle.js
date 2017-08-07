import seedrandom from 'seedrandom';

export function shuffle(rng, array) {
  // Fisher-Yates shuffle, with a provided RNG function.
  // Basically: build up a shuffled part at the end of the array
  // by swapping the last unshuffled element with a random earlier one.
  // See https://bost.ocks.org/mike/shuffle/ for a nice description

  // First, copy the array (bostock's impl forgets this).
  array = Array.prototype.slice.call(array);

  let m = array.length;
  while(m) {
    // Pick an element from the part of the list that's not yet shuffled.
    let prevElement = Math.floor(rng() * m--);

    // Swap it with the current element.
    let tmp = array[prevElement];
    array[prevElement] = array[m];
    array[m] = tmp;
  }
  return array;
}

export function seededShuffle(seed, array) {
  return shuffle(seedrandom(seed), array);
}
