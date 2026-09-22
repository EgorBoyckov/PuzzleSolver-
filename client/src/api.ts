const BASE = "/api/puzzles";

export interface PuzzleStatus {
  id: string;
  has_reference: boolean;
  total_pieces: number;
  batches_uploaded: number;
  located: number | null;
  kind_counts: Record<string, number>;
  steps_total: number;
  steps_pending: number;
}

export interface AssemblyStep {
  step_number: number;
  piece_a: string;
  piece_b: string;
  side_a: number;
  side_b: number;
  rotation_deg: number;
  sector: string | null;
  confidence: number;
  status: "pending" | "done" | "rejected" | "not_found";
}

export interface BatchUploadResult {
  batch_number: number;
  accepted: boolean;
  rejection_reason: string | null;
  pieces_found_in_batch: number;
  pieces_total: number;
  steps_total: number;
}

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* тело не JSON — оставляем statusText */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function createPuzzle(
  puzzleId: string,
  referenceFile: File | null,
  gridRows: number | null,
  gridCols: number | null,
): Promise<void> {
  const form = new FormData();
  form.append("puzzle_id", puzzleId);
  if (gridRows != null) form.append("grid_rows", String(gridRows));
  if (gridCols != null) form.append("grid_cols", String(gridCols));
  if (referenceFile) form.append("reference", referenceFile);
  const res = await fetch(BASE, { method: "POST", body: form });
  await unwrap(res);
}

export async function uploadBatch(puzzleId: string, blob: Blob): Promise<BatchUploadResult> {
  const form = new FormData();
  form.append("file", blob, `frame_${Date.now()}.jpg`);
  const res = await fetch(`${BASE}/${encodeURIComponent(puzzleId)}/batches`, { method: "POST", body: form });
  return unwrap<BatchUploadResult>(res);
}

export async function getStatus(puzzleId: string): Promise<PuzzleStatus> {
  const res = await fetch(`${BASE}/${encodeURIComponent(puzzleId)}`);
  return unwrap<PuzzleStatus>(res);
}

export async function getSteps(puzzleId: string, limit = 10): Promise<AssemblyStep[]> {
  const res = await fetch(`${BASE}/${encodeURIComponent(puzzleId)}/steps?limit=${limit}`);
  return unwrap<AssemblyStep[]>(res);
}

export async function sendFeedback(
  puzzleId: string,
  stepNumber: number,
  status: "done" | "rejected",
): Promise<AssemblyStep[]> {
  const res = await fetch(`${BASE}/${encodeURIComponent(puzzleId)}/steps/${stepNumber}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  return unwrap<AssemblyStep[]>(res);
}
