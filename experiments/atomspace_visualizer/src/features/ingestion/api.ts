import { apiRequest } from '../../shared/api/http';

export interface IngestionResponse {
  status: string;
  message?: string;
  [key: string]: unknown;
}

export const ingestUserData = (username: string) =>
  apiRequest<IngestionResponse>('/api/ingest', {
    method: 'POST',
    body: { username },
  });
