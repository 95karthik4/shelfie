/**
 * Django API client. Native fetch + FormData only -- no HTTP library.
 *
 * The base URL is configuration, never a literal: EXPO_PUBLIC_API_BASE_URL is
 * inlined by Expo at build time from mobile/.env (gitignored). A phone cannot
 * reach the Mac's localhost, so a device demo points this at the Mac's LAN IP.
 */

export const API_BASE_URL = (
  process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

/**
 * A real scan runs a CPU detector and one hosted VLM call; a measured run took
 * 21s. This ceiling exists only so a dead server eventually fails instead of
 * hanging forever -- it is deliberately far above the expected duration.
 */
const SCAN_TIMEOUT_MS = 180_000;
const QUICK_TIMEOUT_MS = 20_000;

export type ItemStatus = 'auto' | 'review' | 'unmatched';

export interface ScanItem {
  id: number;
  index: number;
  raw_title: string | null;
  raw_author: string | null;
  legible: boolean;
  catalog_id: string | null;
  matched_title: string | null;
  matched_author: string | null;
  confidence: number;
  status: ItemStatus;
  reasons: string[];
  confirmed: boolean;
}

export interface ScanResponse {
  scan_id: number;
  detector: { source: string; quality: number; used_fallback: boolean };
  vlm: { latency_ms: number | null; requests_made: number; model: string | null };
  items: ScanItem[];
}

export interface ConfirmedBook {
  id: number;
  scan_item_id: number;
  catalog_id: string | null;
  title: string;
  author: string | null;
  confirmed_at: string;
}

/** Anything the server said no to, in a shape the UI can branch on. */
export class ApiError extends Error {
  status: number;
  code?: string;
  retryable: boolean;

  constructor(message: string, status: number, code?: string, retryable = false) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

/**
 * Turn a failed response into an ApiError.
 *
 * The backend speaks two error shapes: its own {"error": {...}} envelope for
 * VLM/conflict failures, and DRF's {"field": ["message"]} for validation. Both
 * are flattened to a sentence the user can act on.
 */
async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    return new ApiError(
      `Server returned ${response.status} with an unreadable body.`,
      response.status
    );
  }

  if (body && typeof body === 'object' && 'error' in body) {
    const envelope = (body as { error: { code?: string; message?: string; retryable?: boolean } })
      .error;
    return new ApiError(
      envelope?.message ?? `Request failed (${response.status}).`,
      response.status,
      envelope?.code,
      Boolean(envelope?.retryable)
    );
  }

  if (body && typeof body === 'object') {
    const parts: string[] = [];
    for (const [field, value] of Object.entries(body as Record<string, unknown>)) {
      const text = Array.isArray(value) ? value.join(' ') : String(value);
      parts.push(field === 'non_field_errors' || field === 'detail' ? text : `${field}: ${text}`);
    }
    if (parts.length) {
      return new ApiError(parts.join('\n'), response.status);
    }
  }

  return new ApiError(`Request failed (${response.status}).`, response.status);
}

async function request(path: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
  } catch (error) {
    // Network-level failure: wrong IP, server down, phone on another network.
    const aborted = error instanceof Error && error.name === 'AbortError';
    throw new ApiError(
      aborted
        ? 'The server took too long to respond. It may still be processing — check the server log before retrying.'
        : `Could not reach the server at ${API_BASE_URL}.\n\nCheck that Django is running with 0.0.0.0:8000 and that this device is on the same Wi-Fi.`,
      0,
      aborted ? 'timeout' : 'network',
      true
    );
  } finally {
    clearTimeout(timer);
  }
}

/** Guess a filename and MIME type from a local file URI. */
function fileMetaFor(uri: string): { name: string; type: string } {
  const cleaned = uri.split('?')[0];
  const extension = (cleaned.split('.').pop() ?? 'jpg').toLowerCase();
  const type = extension === 'png' ? 'image/png' : 'image/jpeg';
  const name = `shelf.${extension === 'png' ? 'png' : 'jpg'}`;
  return { name, type };
}

export async function uploadScan(imageUri: string): Promise<ScanResponse> {
  const form = new FormData();
  const { name, type } = fileMetaFor(imageUri);
  // React Native's FormData takes this {uri,name,type} shape; the DOM typing
  // does not describe it, hence the cast.
  form.append('photo', { uri: imageUri, name, type } as unknown as Blob);

  const response = await request(
    '/api/scans/',
    { method: 'POST', body: form, headers: { Accept: 'application/json' } },
    SCAN_TIMEOUT_MS
  );
  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as ScanResponse;
}

export type ConfirmPayload = { catalog_id: string } | { title: string; author?: string | null };

export async function confirmScanItem(
  itemId: number,
  payload: ConfirmPayload
): Promise<ConfirmedBook> {
  const response = await request(
    `/api/scan-items/${itemId}/confirm/`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
    },
    QUICK_TIMEOUT_MS
  );
  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as ConfirmedBook;
}

export async function fetchLibrary(): Promise<ConfirmedBook[]> {
  const response = await request(
    '/api/library/',
    { method: 'GET', headers: { Accept: 'application/json' } },
    QUICK_TIMEOUT_MS
  );
  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as ConfirmedBook[];
}
