export function viewIdOf(chunkId: string): string {
  const hash = chunkId.lastIndexOf("#");
  return hash === -1 ? chunkId : chunkId.slice(0, hash);
}

export function chunkId(viewId: string, index: number): string {
  return `${viewId}#${index}`;
}
