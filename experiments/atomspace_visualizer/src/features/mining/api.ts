import { apiRequest } from '../../shared/api/http';

export interface PatternResult {
  pattern: string;
  support: string;
}

export interface MineResponse {
  jobId: string;
  status: 'finished' | 'no_results' | 'error' | string;
  conjunction_count?: number;
  min_support?: number;
  message?: string;
  result?: PatternResult[];
}

export const minePatterns = (conjunctionCount: number, minSupport: number) =>
  apiRequest<MineResponse>('/api/mine', {
    method: 'POST',
    body: {
      conjunction_count: conjunctionCount,
      min_support: minSupport,
    },
  });
