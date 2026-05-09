import { createMemo } from 'solid-js';
import type { GraphData, Triple } from '../../../types';
import { env, isEnabledNumber } from '../../../shared/config/env';
import { MettaParserImpl } from '../../../services/parser/MettaParser';
import { ColumnarTransformer } from '../../../services/graph/ColumnarTransformer';

const emptyGraphData = (): GraphData => ({
  nodes: [],
  edges: [],
  metadata: {
    nodeCount: 0,
    edgeCount: 0,
    hypergraphCount: 0,
    lastUpdated: new Date(),
  },
  hypergraphs: [],
});

const limitTriplesForVisualization = (triples: Triple[]) => {
  if (!isEnabledNumber(env.maxVisualizationArticles)) {
    return triples;
  }

  const allowedArticles = new Set<string>();
  const limited: Triple[] = [];

  for (const triple of triples) {
    const subjects = Array.isArray(triple.subject) ? triple.subject : [triple.subject];
    const articleId = subjects[0];

    if (allowedArticles.has(articleId)) {
      limited.push(triple);
      continue;
    }

    if (allowedArticles.size < env.maxVisualizationArticles) {
      allowedArticles.add(articleId);
      limited.push(triple);
    }
  }

  return limited;
};

const buildVisualizationSubset = (mettaText: string) => {
  if (!isEnabledNumber(env.maxVisualizationArticles)) {
    return mettaText;
  }

  const selectedLines: string[] = [];
  const allowedArticles = new Set<string>();

  for (const rawLine of mettaText.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith(';')) {
      continue;
    }

    const match = line.match(/^\(\s*(?:\(\s*)?[^\s()]+\s+([^\s()]+)\s+/);
    if (!match) {
      continue;
    }

    const articleId = match[1];
    if (!allowedArticles.has(articleId)) {
      if (allowedArticles.size >= env.maxVisualizationArticles) {
        continue;
      }
      allowedArticles.add(articleId);
    }

    selectedLines.push(rawLine);
  }

  return selectedLines.join('\n');
};

export const useVisualizationData = (mettaText: () => string) => {
  const parser = new MettaParserImpl();
  const columnarTransformer = new ColumnarTransformer();

  return createMemo<GraphData>(() => {
    const source = mettaText().trim();
    if (!source) {
      return emptyGraphData();
    }

    try {
      const visualizationText = buildVisualizationSubset(source);
      const triples = parser.extractTriples(visualizationText);
      const visualizationTriples = limitTriplesForVisualization(triples);
      return columnarTransformer.transformToColumnar(visualizationTriples);
    } catch (error) {
      console.error('Parsing error:', error);
      return emptyGraphData();
    }
  });
};
