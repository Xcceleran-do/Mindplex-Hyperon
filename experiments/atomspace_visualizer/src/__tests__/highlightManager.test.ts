import { describe, expect, it } from 'vitest';
import type { GraphData, GraphNode } from '../types';
import { updateHighlightState } from '../utils/highlightManager';
import { MettaParserImpl } from '../services/parser/MettaParser';
import { ColumnarTransformer } from '../services/graph/ColumnarTransformer';

const node = (
  id: string,
  label: string,
  columnType: 'article' | 'property',
  propertyName?: string,
): GraphNode => ({
  id,
  label,
  type: 'entity',
  position: { x: 0, y: 0 },
  metadata: {
    columnType,
    propertyName,
    originalExpression: columnType === 'article' ? label : undefined,
  },
});

const graphData: GraphData = {
  nodes: [
    node('article-a', 'a', 'article'),
    node('article-b', 'b', 'article'),
    node('article-c', 'c', 'article'),
    node('length-short', 'short', 'property', 'length'),
    node('length-long', 'long', 'property', 'length'),
    node('tone-positive', 'positive', 'property', 'tone'),
    node('tone-negative', 'negative', 'property', 'tone'),
  ],
  edges: [
    { id: 'a-short', source: 'article-a', target: 'length-short', label: 'length', directed: true, type: 'relation' },
    { id: 'a-positive', source: 'article-a', target: 'tone-positive', label: 'tone', directed: true, type: 'relation' },
    { id: 'b-short', source: 'article-b', target: 'length-short', label: 'length', directed: true, type: 'relation' },
    { id: 'b-negative', source: 'article-b', target: 'tone-negative', label: 'tone', directed: true, type: 'relation' },
    { id: 'c-long', source: 'article-c', target: 'length-long', label: 'length', directed: true, type: 'relation' },
    { id: 'c-positive', source: 'article-c', target: 'tone-positive', label: 'tone', directed: true, type: 'relation' },
  ],
  metadata: { nodeCount: 7, edgeCount: 6, hypergraphCount: 0, lastUpdated: new Date() },
  hypergraphs: [],
};

describe('updateHighlightState', () => {
  it('requires an article to match every selected attribute', () => {
    const result = updateHighlightState(graphData, {
      active: true,
      propertyFilters: [
        { property: 'length', value: 'short' },
        { property: 'tone', value: 'positive' },
      ],
    });

    expect(result.highlightedNodes.has('article-a')).toBe(true);
    expect(result.highlightedNodes.has('article-b')).toBe(false);
    expect(result.highlightedNodes.has('article-c')).toBe(false);
    expect(result.highlightedEdges).toEqual(new Set(['a-short', 'a-positive']));
  });

  it('does not treat multiple values for one attribute as alternatives', () => {
    const result = updateHighlightState(graphData, {
      active: true,
      propertyFilters: [
        { property: 'tone', value: 'positive' },
        { property: 'tone', value: 'negative' },
      ],
    });

    expect([...result.highlightedNodes].filter((id) => id.startsWith('article-'))).toEqual([]);
    expect(result.highlightedEdges.size).toBe(0);
  });

  it('combines an article selection with all attribute filters', () => {
    const result = updateHighlightState(graphData, {
      active: true,
      articleIds: ['b'],
      propertyFilters: [{ property: 'tone', value: 'positive' }],
    });

    expect([...result.highlightedNodes].filter((id) => id.startsWith('article-'))).toEqual([]);
  });

  it('intersects filters produced from real PeTTa-formatted facts', () => {
    const source = [
      '((engagement A_1 "High") (STV 0.8 0.9))',
      '((tone A_1 "Analytical") (STV 0.9 0.9))',
      '((engagement A_2 "High") (STV 0.8 0.9))',
      '((tone A_2 "Critical") (STV 0.9 0.9))',
      '((engagement A_3 "Low") (STV 0.3 0.9))',
      '((tone A_3 "Analytical") (STV 0.9 0.9))',
    ].join('\n');
    const triples = new MettaParserImpl().extractTriples(source);
    const parsedGraph = new ColumnarTransformer().transformToColumnar(triples);

    const result = updateHighlightState(parsedGraph, {
      active: true,
      propertyFilters: [
        { property: 'engagement', value: '"High"' },
        { property: 'tone', value: '"Analytical"' },
      ],
    });

    expect(
      parsedGraph.nodes
        .filter(
          (candidate) => candidate.metadata.columnType === 'article'
            && result.highlightedNodes.has(candidate.id),
        )
        .map((candidate) => candidate.metadata.originalExpression),
    ).toEqual(['A_1']);
  });
});
