export default function countWords(str) {
    var matches = str.match(/\S+/g);
    return matches ? matches.length : 0;
}
