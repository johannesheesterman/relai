export function precisionAtK(retrieved: string[], relevant: string[], k: number): number {
  const top = retrieved.slice(0, k);
  if (top.length === 0) return 0;
  const rel = new Set(relevant);
  const hits = top.filter((id) => rel.has(id)).length;
  return hits / top.length;
}

export function recallAtK(retrieved: string[], relevant: string[], k: number): number {
  if (relevant.length === 0) return 0;
  const top = new Set(retrieved.slice(0, k));
  const hits = relevant.filter((id) => top.has(id)).length;
  return hits / relevant.length;
}

export function f1AtK(retrieved: string[], relevant: string[], k: number): number {
  const p = precisionAtK(retrieved, relevant, k);
  const r = recallAtK(retrieved, relevant, k);
  if (p + r === 0) return 0;
  return (2 * p * r) / (p + r);
}
