import { env } from '../config/env';

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

type JsonRequestOptions = Omit<RequestInit, 'body' | 'headers'> & {
  body?: unknown;
  headers?: HeadersInit;
};

const toUrl = (path: string) => `${env.apiBaseUrl}${path}`;

const parseResponse = async (response: Response) => {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    return response.json();
  }

  return response.text();
};

const fallbackMessage = (status: number) => {
  if (status === 400) return 'The request is invalid. Check the entered values and try again.';
  if (status === 401) return 'Your session has expired. Sign in again and retry.';
  if (status === 403) return 'You do not have permission to perform this action.';
  if (status === 404) return 'The requested service could not be found.';
  if (status === 408 || status === 504) return 'The request timed out. Try again in a moment.';
  if (status === 429) return 'The service is busy. Wait briefly and try again.';
  if (status >= 500) return 'The service could not complete the request. Try again later.';
  return 'The request could not be completed.';
};

const errorMessage = (payload: unknown, status: number) => {
  if (!payload || typeof payload !== 'object') return fallbackMessage(status);
  const candidate = payload as { message?: unknown; error?: unknown };
  if (typeof candidate.message === 'string' && candidate.message.trim()) return candidate.message;
  if (candidate.error && typeof candidate.error === 'object') {
    const nestedMessage = (candidate.error as { message?: unknown }).message;
    if (typeof nestedMessage === 'string' && nestedMessage.trim()) return nestedMessage;
  }
  if (typeof candidate.error === 'string' && candidate.error.trim()) return candidate.error;
  return fallbackMessage(status);
};

export const apiRequest = async <T>(path: string, options: JsonRequestOptions = {}): Promise<T> => {
  const response = await fetch(toUrl(path), {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });

  const payload = await parseResponse(response);

  if (!response.ok) {
    throw new ApiError(errorMessage(payload, response.status), response.status, payload);
  }

  return payload as T;
};

export const fetchText = async (path: string): Promise<string> => {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await parseResponse(response);
    throw new ApiError(errorMessage(payload, response.status), response.status, payload);
  }
  return response.text();
};
