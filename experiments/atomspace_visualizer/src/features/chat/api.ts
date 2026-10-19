import { apiRequest } from '../../shared/api/http';
import type { PatternResult } from '../mining/api';

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatHistoryMessage {
  role: ChatRole;
  content: string;
}

export interface ChatResponse {
  response: string;
  functionCalls?: unknown;
  session_id?: string;
}

export interface PatternSummaryResponse {
  summary: string;
}

export const summarizePatterns = (patterns: PatternResult[]) =>
  apiRequest<PatternSummaryResponse>('/api/chat/summarize', {
    method: 'POST',
    body: { patterns },
  });

export const analyzePattern = (pattern: string, support: string) =>
  apiRequest<PatternSummaryResponse>('/api/chat/analyze', {
    method: 'POST',
    body: { pattern, support },
  });

export const sendChatMessage = (message: string, history: ChatHistoryMessage[], sessionId = 'default') =>
  apiRequest<ChatResponse>('/api/chat', {
    method: 'POST',
    body: {
      message,
      history,
      session_id: sessionId,
    },
  });

export const clearChatSession = (sessionId = 'default') =>
  apiRequest<{ status: string }>('/api/chat/clear', {
    method: 'POST',
    body: { session_id: sessionId },
  });
