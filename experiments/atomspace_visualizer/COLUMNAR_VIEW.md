# Columnar Property-Based Visualization (Unified Guide)

This document merges and streamlines the previous columnar docs into one guide. It explains how the columnar mode works, how to use it, and how it’s implemented in the AtomSpace Visualizer.

## Overview

The columnar view transforms property-based MeTTa data into an easy-to-scan layout:

- Articles column (first): nodes for each article/entity (0, 1, 2, ...)
- Property columns: one column per property (topic, length, tone, writing_style, engagement_level, ...)
- Property value nodes: unique values under each property column
- “None” nodes: handle missing values at the bottom of each property column
- Continuous lines: each article has ONE smooth curved path through all its property values

## Features

1) Columnar layout
- Clean structure that scales to larger datasets with a scrollable viewport
- Column separators for readability

2) Interactive filtering (single and multi-select)
- Click an article to highlight that article and all its properties
- Click a property value to highlight all articles having that value
- Hold Ctrl/Cmd to multi-select articles and/or properties
- Active filters highlighted in amber; others are dimmed
- Click empty space (or use Clear All) to reset

3) Clickable, collapsible legend
- Starts collapsed; expand to filter via article list and property-grouped values
- Displays active filter summary with a Clear button

4) Collapsible UI controls
- Zoom in/out and reset view
- Built-in quick instructions

5) Modern design and performance
- Glass-morphism cards, gradients, smooth animations
- Canvas-based renderer with viewport culling for 60fps-level performance
- Tested on modern browsers (Chrome, Edge, Firefox, Safari)

## Usage

Navigation
- Pan: click-drag on empty space
- Zoom: mouse wheel
- Reset: use the reset button in the controls

Filtering
- Single select: click any article or property value
- Multi-select: hold Ctrl/Cmd while clicking
- Mixed filters: combine multiple articles and properties at once
- Clear: click empty canvas space or Clear in the legend

## Data Format

MeTTa input is property-based triples in this shape:

```metta
(property article_id "value")
```

Example:
```metta
(topic 0 "AI")
(length 0 "high")
(tone 0 "Analytical")
(writing_style 0 "Formal")
(engagement_level 0 "high")

(topic 1 "Gardening")
(length 1 "low")
...
```

Where:
- First argument: property name (becomes column header)
- Second argument: article id (rendered in the first column)
- Third argument: property value (becomes a node in that property’s column)

## Mining integration (optional)

The columnar app includes a Mining Panel that can visualize mined patterns by highlighting matching property values and connected articles.

- Mining endpoint: http://localhost:8000/api/mine
- Results are displayed in a collapsible card; click “Visualize” to apply filters corresponding to the pattern’s property=value pairs

## Architecture & Files

Key components
- AppColumnar.tsx: app entry for columnar mode
- components/ColumnarVisualizer/ColumnarVisualizer.tsx: canvas renderer with continuous per-article lines and highlighting
- services/graph/ColumnarTransformer.ts: turns MeTTa triples into columnar GraphData (articles, headers, values, edges)
- components/Legend/EnhancedLegend.tsx: clickable, collapsible legend with active filter summary
- components/UIControls/EnhancedUIControls.tsx: zoom/reset and instructions card
- types/index.ts: includes FilterState for multi-select (articleIds and propertyFilters)

Data flow
1. MeTTa text → MettaParser → Triples
2. ColumnarTransformer → GraphData (nodes/edges with column positions)
3. ColumnarVisualizer (Canvas) → rendering and interaction
4. FilterState drives highlighting and dimming

Performance & compatibility
- Canvas with viewport culling to avoid offscreen work
- Smooth animations; handles 100+ nodes comfortably on modern hardware
- Works on Chrome, Edge, Firefox, Safari

## Switching between views

```typescript
import AppColumnar from './AppColumnar';
render(() => <AppColumnar />, root!);
```

## Future enhancements

- Export filtered view as image
- Search
- More complex filter logic (AND/OR)
- Property statistics in legend
- Theming / dark mode
- Animated transitions on filter changes

## Quick run (dev)

```bash
cd experiments/atomspace_visualizer
npm install
npm run dev
# Open http://localhost:3000
```
