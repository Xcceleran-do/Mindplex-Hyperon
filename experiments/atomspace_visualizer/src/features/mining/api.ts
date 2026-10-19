import { apiRequest } from '../../shared/api/http';

export interface PatternResult {
  pattern: string;
  support: string;
}

export interface MineResponse {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'finished' | 'no_results' | 'error' | string;
  conjunction_count?: number;
  min_support?: number;
  message?: string;
  result?: PatternResult[];
  inserted_rules?: string[];
  rule_insertion?: unknown;
  error?: string;
}

interface MiningStatusResponse {
  jobId: string;
  status: string;
  conjunction_count?: number;
  min_support?: number;
  message?: string;
  result?: {
    status?: string;
    message?: string;
    patterns?: PatternResult[];
    rules?: string[];
    rule_insertion?: unknown;
  };
  error?: string;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const normalizeMiningStatus = (payload: MiningStatusResponse): MineResponse => {
  if (payload.status === 'completed') {
    const innerStatus = payload.result?.status;
    if (innerStatus === 'no_results') {
      return {
        ...payload,
        status: 'no_results',
        message: payload.message || payload.result?.message,
        result: [],
      };
    }

    return {
      ...payload,
      status: innerStatus === 'success' ? 'finished' : payload.status,
      message: payload.message || payload.result?.message,
      result: payload.result?.patterns || [],
      inserted_rules: payload.result?.rules || [],
      rule_insertion: payload.result?.rule_insertion,
    };
  }

  return {
    ...payload,
    result: [],
  };
};

const pollMiningJob = async (jobId: string): Promise<MineResponse> => {
  const startedAt = Date.now();
  const timeoutMs = 5 * 60 * 1000;

  while (Date.now() - startedAt < timeoutMs) {
    const status = await apiRequest<MiningStatusResponse>(`/api/mine/${jobId}`);
    const normalized = normalizeMiningStatus(status);

    if (normalized.status === 'finished' || normalized.status === 'no_results') {
      return normalized;
    }

    if (normalized.status === 'error') {
      throw new Error(normalized.error || normalized.message || 'Mining failed.');
    }

    await sleep(1500);
  }

  throw new Error('Mining timed out while waiting for the backend job.');
};

export const minePatterns = async (conjunctionCount: number, minSupport: number) => {
  const started = await apiRequest<MineResponse>('/api/mine', {
    method: 'POST',
    body: {
      conjunction_count: conjunctionCount,
      min_support: minSupport,
    },
  });

  if (started.status === 'finished' || started.status === 'no_results') {
    return started;
  }

  if (!started.jobId) {
    throw new Error(started.message || 'Mining did not return a job id.');
  }

  return pollMiningJob(started.jobId);
};
