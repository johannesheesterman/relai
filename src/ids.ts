export function viewIdOf(chunkId: string): string {
  // Strip only a trailing chunk suffix `#<digits>`, so view ids whose remoteId
  // legitimately contains '#' are not mis-collapsed during fusion grouping.
  return chunkId.replace(/#\d+$/, "");
}

export function chunkId(viewId: string, index: number): string {
  return `${viewId}#${index}`;
}
