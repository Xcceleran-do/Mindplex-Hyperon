import { apiRequest } from '../../shared/api/http';

export interface SimulationAttribute {
  value: string;
  strength?: number;
  confidence?: number;
}

export interface SimulationPayload {
  article_id?: string;
  attributes: Record<string, string | SimulationAttribute>;
  depth?: number;
}

export interface SimulationBucket {
  proofs: string[];
  proof_count: number;
  aggregated_stv: {
    strength: number;
    confidence: number;
  } | null;
  conditional_stv?: {
    strength: number;
    confidence: number;
  } | null;
  conditional_score?: number;
  conditional_suggestions?: SimulationConditionalSuggestion[];
  exact_score?: number;
  raw_score: number;
  probability: number;
}

export interface SimulationFactDetail {
  id: string;
  atom: string;
  strength: number | null;
  confidence: number | null;
}

export interface SimulationRuleAtom {
  atom: string;
  predicate: string;
  subject: string;
  value: string;
}

export interface SimulationRuleDetail {
  id: string;
  atom: string;
  antecedents: SimulationRuleAtom[];
  consequent: SimulationRuleAtom | null;
  stv: {
    strength: number;
    confidence: number;
  } | null;
}

export interface SimulationProofChain {
  proof: string;
  rule_id: string;
  rule: SimulationRuleDetail | null;
  facts: SimulationFactDetail[];
  stv: {
    strength: number;
    confidence: number;
  } | null;
}

export interface SimulationUnmatchedRule {
  rule_id: string;
  consequent: SimulationRuleAtom | null;
  matched_antecedents: Array<{
    required: string;
    fact: SimulationFactDetail;
  }>;
  missing_antecedents: string[];
  rule: SimulationRuleDetail;
}

export interface SimulationConditionalSuggestion extends SimulationUnmatchedRule {
  assumed_antecedents: string[];
  conditional_stv: {
    strength: number;
    confidence: number;
  };
  conditional_score: number;
  matched_count: number;
  missing_count: number;
  summary: string;
}

export interface SimulationExplanation {
  summary: string;
  input_facts: SimulationFactDetail[];
  rules: SimulationRuleDetail[];
  chains_by_level: Record<'High' | 'Medium' | 'Low', SimulationProofChain[]>;
  unmatched_rules: SimulationUnmatchedRule[];
  conditional_suggestions_by_level?: Record<'High' | 'Medium' | 'Low', SimulationConditionalSuggestion[]>;
  conditional_suggestions?: SimulationConditionalSuggestion[];
}

export interface SimulationResponse {
  status: 'success' | 'error' | string;
  article_id: string;
  depth_used: number;
  rules_used: number;
  used_prior_fallback: boolean;
  proof_count: number;
  input_facts: string[];
  probabilities: Record<'High' | 'Medium' | 'Low', number>;
  predicted_engagement: 'High' | 'Medium' | 'Low' | string;
  buckets: Record<'High' | 'Medium' | 'Low', SimulationBucket>;
  explanation?: SimulationExplanation;
  message?: string;
}

export const simulateEngagement = (payload: SimulationPayload) =>
  apiRequest<SimulationResponse>('/api/simulate', {
    method: 'POST',
    body: payload,
  });
