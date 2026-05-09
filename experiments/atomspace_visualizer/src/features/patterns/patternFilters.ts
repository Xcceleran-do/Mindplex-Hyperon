import type { FilterState } from '../../types';

export const filterStateFromPattern = (pattern: string): FilterState | null => {
  const propertyFilters: Array<{ property: string; value: string }> = [];
  const regex = /\(([^\s()]+)\s+\$[^\s()]+\s+("[^"]+")\)/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(pattern)) !== null) {
    propertyFilters.push({
      property: match[1],
      value: match[2],
    });
  }

  if (propertyFilters.length === 0) {
    return null;
  }

  return {
    active: true,
    articleIds: [],
    propertyFilters,
  };
};
