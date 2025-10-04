# Columnar Property-Based Visualization

This is an enhanced visualization mode for the AtomSpace Visualizer that displays property-based data in a columnar layout.

## Overview

The columnar visualization transforms property-based Metta data (like article properties) into an intuitive column-based layout where:

- **First Column (Articles)**: Contains nodes for each article/entity (0, 1, 2, 3, ...)
- **Property Columns**: Each unique property type (topic, length, tone, writing_style, engagement_level) gets its own column
- **Property Values**: Under each column are nodes representing the unique values that property can take
- **None Nodes**: At the bottom of each property column is a "None" node for handling missing values
- **Connections**: Lines connect each article to its corresponding property values

## Features

### 1. Columnar Layout
- Clean, organized presentation of relationships
- Easy to scan and compare properties across articles
- Scalable for large datasets with scrollable viewport

### 2. Interactive Filtering
- **Click on Articles**: Highlight an article and all its property connections
- **Click on Property Values**: Highlight all articles that have that property value
- **Color Contrast**: Filtered items are highlighted in amber/gold, while non-filtered items are dimmed
- **Persistent Filtering**: Filter remains active until cleared or changed

### 3. Clickable Legend
- **Collapsible by Default**: Legend starts collapsed to save space
- **Node Type Explanations**: Shows what each node type represents
- **Clickable Items**: Click on articles or properties in the legend to filter the visualization
- **Active Filter Status**: Shows which filter is currently active with a clear button

### 4. Enhanced UI Controls
- **Collapsible by Default**: Controls panel starts collapsed
- **Zoom Controls**: Zoom in/out buttons for easy navigation
- **Reset View**: Quickly return to the default view
- **Instructions**: Built-in guide for using the visualization

### 5. Modern Design
- **Smooth Animations**: All interactions have elegant transitions
- **Gradient Backgrounds**: Beautiful color schemes for UI elements
- **Glass Morphism**: Translucent panels with backdrop blur effects
- **Responsive**: Adapts to different screen sizes

## Usage

### Navigation
- **Pan**: Click and drag on empty space to move the view
- **Zoom**: Scroll wheel to zoom in/out
- **Select**: Click on nodes to filter by them

### Filtering
1. Click on any article node (in the first column) to see all its properties
2. Click on any property value node to see all articles with that value
3. Use the legend to quickly filter by clicking on articles or values
4. Click "Clear" in the filter status banner to remove the filter

### UI Elements
- **Legend** (bottom-left): Shows node types and provides filtering controls
- **Controls** (top-right): Provides zoom and view controls
- **Title Bar** (top-center): Shows graph statistics

## Data Format

The visualization expects Metta data in this format:

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
- First argument is the property name
- Second argument is the article ID
- Third argument is the property value

## Implementation Details

### Key Files
- `AppColumnar.tsx`: Main application component for columnar view
- `ColumnarVisualizer.tsx`: Canvas-based columnar visualization component
- `ColumnarTransformer.ts`: Transforms triples into columnar layout
- `EnhancedLegend.tsx`: Interactive, collapsible legend with filtering
- `EnhancedUIControls.tsx`: Modern, collapsible control panel

### Architecture
1. **Data Flow**: Metta → Parser → Triples → Columnar Transformer → Graph Data
2. **Rendering**: Graph Data → Canvas Renderer → Interactive Visualization
3. **State Management**: SolidJS signals for reactive updates
4. **Filtering**: FilterState propagates from legend/clicks to visualization

## Switching Between Views

To switch between the original and columnar views:

**Columnar View** (current):
```typescript
// src/index.tsx
import AppColumnar from './AppColumnar';
render(() => <AppColumnar />, root!);
```

**Original View**:
```typescript
// src/index.tsx
import App from './App';
render(() => <App />, root!);
```

## Future Enhancements

- [ ] Export filtered view as image
- [ ] Multiple simultaneous filters
- [ ] Search functionality
- [ ] Custom color schemes
- [ ] Animated transitions between filter states
- [ ] Property statistics in legend
- [ ] Grouping by property values

## Performance

- Optimized canvas rendering with culling
- Only visible nodes are rendered
- Smooth 60fps animations
- Handles 100+ nodes efficiently
