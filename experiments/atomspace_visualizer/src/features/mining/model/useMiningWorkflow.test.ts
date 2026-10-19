import { describe, expect, it } from 'vitest';
import { sortPatternsBySupport } from './useMiningWorkflow';

describe('sortPatternsBySupport', () => {
  it('sorts numeric support descending without mutating the API result', () => {
    const patterns = [
      { pattern: 'rule-b', support: '2' },
      { pattern: 'rule-a', support: '10' },
      { pattern: 'rule-c', support: '4.5' },
    ];

    expect(sortPatternsBySupport(patterns).map((pattern) => pattern.support)).toEqual(['10', '4.5', '2']);
    expect(patterns.map((pattern) => pattern.support)).toEqual(['2', '10', '4.5']);
  });

  it('places missing support last and uses pattern text for stable ties', () => {
    const patterns = [
      { pattern: 'rule-z', support: '3' },
      { pattern: 'rule-a', support: 'unknown' },
      { pattern: 'rule-b', support: '3' },
    ];

    expect(sortPatternsBySupport(patterns).map((pattern) => pattern.pattern)).toEqual([
      'rule-b',
      'rule-z',
      'rule-a',
    ]);
  });
});
