import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, extname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const srcRoot = join(testDir, '..');

const sourceExtensions = new Set(['.css', '.ts', '.tsx', '.js', '.jsx', '.mjs']);
const ignoredPathParts = new Set(['__tests__', 'dist', 'node_modules']);

const collectSourceFiles = (directory) => {
  const files = [];

  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    const stats = statSync(path);

    if (stats.isDirectory()) {
      if (!ignoredPathParts.has(entry)) {
        files.push(...collectSourceFiles(path));
      }
      continue;
    }

    if (sourceExtensions.has(extname(path))) {
      files.push(path);
    }
  }

  return files;
};

const sourceFiles = collectSourceFiles(srcRoot);

const readSources = () =>
  sourceFiles.map((path) => ({
    path: relative(srcRoot, path).split(sep).join('/'),
    content: readFileSync(path, 'utf8'),
  }));

const findMatches = (patterns) =>
  readSources().flatMap(({ path, content }) => {
    const lines = content.split(/\r?\n/);
    return lines.flatMap((line, index) =>
      patterns
        .filter((pattern) => pattern.test(line))
        .map((pattern) => `${path}:${index + 1} matched ${pattern}: ${line.trim()}`),
    );
  });

const themedFiles = [
  'AppColumnar.module.css',
  'components/ChatInterface/ChatInterface.css',
  'components/MiningInterface/MiningInterface.css',
  'components/Legend/EnhancedLegend.module.css',
  'features/visualization/atlas/SemanticAtlas.module.css',
  'features/visualization/atlas/SemanticAtlas.tsx',
];

const readThemedSources = () =>
  themedFiles.map((path) => ({
    path,
    content: readFileSync(join(srcRoot, path), 'utf8'),
  }));

const findThemedMatches = (patterns) =>
  readThemedSources().flatMap(({ path, content }) => {
    const lines = content.split(/\r?\n/);
    return lines.flatMap((line, index) =>
      patterns
        .filter((pattern) => pattern.test(line))
        .map((pattern) => `${path}:${index + 1} matched ${pattern}: ${line.trim()}`),
    );
  });

describe('color system', () => {
  it('does not use gradients or background images', () => {
    const matches = findMatches([
      /\b(?:linear|radial|conic|repeating-linear|repeating-radial)-gradient\s*\(/i,
      /create(?:Linear|Radial)Gradient\s*\(/,
      /\bbackground-image\s*:/i,
    ]);

    expect(matches).toEqual([]);
  });

  it('does not reintroduce the old prototype palette', () => {
    const matches = findMatches([
      /#667eea/i,
      /#764ba2/i,
      /#5568d3/i,
      /#6a3f91/i,
      /#4557c2/i,
      /#5a3580/i,
      /#3b82f6/i,
      /#1d4ed8/i,
      /#06b6d4/i,
      /#8b5cf6/i,
      /#ec4899/i,
      /rgba\(\s*102\s*,\s*126\s*,\s*234/i,
      /rgba\(\s*139\s*,\s*92\s*,\s*246/i,
      /rgba\(\s*6\s*,\s*182\s*,\s*212/i,
      /rgba\(\s*79\s*,\s*70\s*,\s*229/i,
    ]);

    expect(matches).toEqual([]);
  });

  it('keeps the approved flat brand tokens in variables.css', () => {
    const variables = readFileSync(join(srcRoot, 'styles', 'variables.css'), 'utf8');

    expect(variables).toContain('--color-bg: #fafafa;');
    expect(variables).toContain('--color-accent: #00875a;');
    expect(variables).toContain('--color-bg: #09090b;');
    expect(variables).toContain('--color-accent: #3ecf8e;');
  });

  it('keeps primary themed surfaces on semantic color tokens', () => {
    const matches = findThemedMatches([
      /#[0-9a-f]{3,8}\b/i,
      /rgba?\(/i,
      /hsla?\(/i,
    ]);

    expect(matches).toEqual([]);
  });
});
