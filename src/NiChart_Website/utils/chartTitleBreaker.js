// Function to break a string into smaller parts with <br> tags
export function breakStringIntoParts(str, maxChars) {
  const parts = [];
  let currentPart = '';

  for (let i = 0; i < str.length; i++) {
    currentPart += str[i];

    if (str[i] === '|' || currentPart.length >= maxChars) {
      parts.push(currentPart);
      currentPart = '';
    }
  }

  if (currentPart.length > 0) {
    parts.push(currentPart);
  }

  const chunks = parts.reduce((result, part) => {
    if (part.length <= maxChars) {
      result.push(part);
    } else {
      for (let i = 0; i < part.length; i += maxChars) {
        result.push(part.slice(i, i + maxChars));
      }
    }
    return result;
  }, []);

  return chunks.join('<br>');
}
