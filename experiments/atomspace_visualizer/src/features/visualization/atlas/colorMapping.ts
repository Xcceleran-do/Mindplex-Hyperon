export type AtlasColorTheme = {
  article: string;
  palette: string[];
};

export type AtlasColorSubject = {
  kind: 'article' | 'property';
  label: string;
  propertyName?: string;
};

const DEFAULT_PALETTE_SIZE = 8;

export const hashColorKey = (value: string): number => {
  let result = 0;
  for (let index = 0; index < value.length; index += 1) {
    result = value.charCodeAt(index) + ((result << 5) - result);
  }
  return Math.abs(result);
};

const propertyColorKey = (subject: Pick<AtlasColorSubject, 'propertyName' | 'label'>) =>
  subject.propertyName || subject.label;

export const getPaletteIndexForProperty = (
  propertyName: string,
  label = '',
  paletteSize = DEFAULT_PALETTE_SIZE,
): number => {
  const normalizedSize = Math.max(1, paletteSize);
  return hashColorKey(propertyName || label) % normalizedSize;
};

export const colorTokenForAtlasSubject = (subject: AtlasColorSubject, paletteSize = DEFAULT_PALETTE_SIZE): string => {
  if (subject.kind === 'article') {
    return 'var(--node-article)';
  }

  const paletteIndex = getPaletteIndexForProperty(subject.propertyName || '', subject.label, paletteSize);
  return `var(--viz-${paletteIndex + 1})`;
};

const overflowPropertyColor = (propertyName: string, attempt: number): string => {
  const seed = hashColorKey(`${propertyName}:${attempt}`);
  const hue = seed % 360;
  const saturation = 56 + (Math.floor(seed / 360) % 5) * 7;
  const lightness = 42 + (Math.floor(seed / 1800) % 4) * 8;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
};

export const buildPropertyColorMap = (propertyNames: string[], theme: AtlasColorTheme): Map<string, string> => {
  const uniqueProperties = Array.from(new Set(propertyNames.filter(Boolean))).sort((a, b) => a.localeCompare(b));
  const map = new Map<string, string>();
  const usedColors = new Set<string>();
  const palette = theme.palette.filter(Boolean);

  for (const propertyName of uniqueProperties) {
    let assigned: string | undefined;

    if (palette.length > 0) {
      const preferredIndex = hashColorKey(propertyName) % palette.length;
      for (let offset = 0; offset < palette.length; offset += 1) {
        const candidate = palette[(preferredIndex + offset) % palette.length];
        if (!usedColors.has(candidate)) {
          assigned = candidate;
          break;
        }
      }
    }

    if (!assigned) {
      let attempt = 0;
      do {
        assigned = overflowPropertyColor(propertyName, attempt);
        attempt += 1;
      } while (usedColors.has(assigned));
    }

    map.set(propertyName, assigned);
    usedColors.add(assigned);
  }

  return map;
};

export const readAtlasColorTheme = (rootStyle: CSSStyleDeclaration, paletteSize = DEFAULT_PALETTE_SIZE): AtlasColorTheme => ({
  article: rootStyle.getPropertyValue('--node-article').trim(),
  palette: Array.from(
    { length: paletteSize },
    (_, index) => rootStyle.getPropertyValue(`--viz-${index + 1}`).trim(),
  ),
});

export const colorForAtlasSubject = (
  subject: AtlasColorSubject,
  theme: AtlasColorTheme,
  propertyColors?: Map<string, string>,
): string => {
  if (subject.kind === 'article') {
    return theme.article;
  }

  const key = propertyColorKey(subject);
  const explicitColor = propertyColors?.get(key);
  if (explicitColor) {
    return explicitColor;
  }

  return theme.palette[getPaletteIndexForProperty(key, '', theme.palette.length)] || theme.article;
};
