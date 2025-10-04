# AtomSpace Visualizer

An advanced web-based visualization tool for MeTTa knowledge bases, featuring both traditional graph views and a modern columnar property-based visualization mode.

## 🚀 Quick Start

```bash
cd experiments/atomspace_visualizer
npm install
npm run dev
```

Visit: http://localhost:3000/

## 📊 Visualization Modes

### Columnar View (Current Default)
A modern property-based visualization that displays data in an intuitive columnar layout:

- **Articles Column**: All entities (articles 0-9) in the first column
- **Property Columns**: Each property type (topic, length, tone, etc.) gets its own column
- **Continuous Lines**: Each article has ONE smooth curved line flowing through all its property values
- **None Nodes**: Handle missing property values elegantly
- **Multi-Select Filtering**: Hold Ctrl/Cmd to select multiple articles or properties
- **Collapsible UI**: Legend and controls start minimized for a clean interface

### Traditional Graph View
Classic node-link diagram with multiple layout algorithms (force-directed, hierarchical, circular).

## ✨ Key Features

### 1. Interactive Filtering
- **Single Click**: Highlight an article or property value
- **Multi-Select**: Hold **Ctrl** (Windows/Linux) or **Cmd** (Mac) while clicking
- **Visual Feedback**: Selected items highlighted in orange, others dimmed
- **Clear Filter**: Click "Clear All" or click empty space

### 2. Collapsible Panels
- **Legend** (bottom-left): Node types and clickable filters
- **Controls** (top-right): Zoom and view controls
- Both start collapsed to maximize visualization space

### 3. Modern UI/UX
- Smooth animations and transitions
- Gradient backgrounds with glass-morphism effects
- Responsive design for all screen sizes
- Color-coded node types for easy identification

## 🎯 How to Use

### Navigation
- **Pan**: Click and drag on empty space
- **Zoom**: Scroll wheel to zoom in/out
- **Reset**: Click reset button in controls

### Filtering
1. **Single Select**: Click any article node or property value
2. **Multi-Select**: Hold Ctrl/Cmd and click multiple items
3. **Mix Filters**: Select multiple articles AND property values simultaneously
4. **Clear**: Click "Clear All" button or click empty space

### Understanding the Layout
```
┌──────────┬─────────┬─────────┬──────────┬─────────────┐
│ ARTICLES │  TOPIC  │ LENGTH  │   TONE   │    STYLE    │
├──────────┼─────────┼─────────┼──────────┼─────────────┤
│    0 ────┼→ AI     │         │          │             │
│          │         ├→ high   │          │             │
│    1     │         │         ├→ Formal  │             │
│          │         │         │          ├→ Analytical │
│    2     │         ├→ medium │          │             │
└──────────┴─────────┴─────────┴──────────┴─────────────┘
```

Each article has **one continuous curved line** connecting all its properties.

## 📁 Project Structure

### Core Features
- **Real-time MeTTa parsing** with syntax validation
- **Interactive visualizations** with zoom, pan, and filtering
- **Multiple layout modes**: Columnar and traditional graph layouts
- **Pattern mining integration** for discovering frequent patterns
- **Export capabilities** for sharing visualizations

## 🏗️ Architecture

### Key Components

#### Visualization Modes
- **AppColumnar.tsx**: Columnar property-based visualization (current default)
- **App.tsx**: Traditional graph visualization with force-directed layouts

#### Services
- **ColumnarTransformer**: Transforms MeTTa data into columnar layout
- **MettaParser**: Parses MeTTa expressions into structured data
- **GraphTransformer**: Converts triples to graph structures
- **GraphEngine**: Handles layout algorithms and animations

#### Components
- **ColumnarVisualizer**: Canvas-based columnar rendering
- **GraphVisualizer**: Traditional node-link diagram renderer
- **EnhancedLegend**: Collapsible, interactive legend with filtering
- **EnhancedUIControls**: Zoom and view controls
- **MiningPanel**: Pattern mining interface
- **MettaEditor**: Code editor for MeTTa expressions

## 💾 Data Format

The columnar visualization expects MeTTa data in this format:

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
(tone 1 "Casual")
...
```

Where:
- **First argument**: Property name (becomes a column header)
- **Second argument**: Article ID (shown in first column)
- **Third argument**: Property value (node in that property's column)

## 🎨 Color Scheme

- **Articles**: Blue (#3b82f6)
- **Property Headers**: Purple (#8b5cf6) 
- **Property Values**: Green (#10b981)
- **Missing Values (None)**: Gray (#6b7280)
- **Highlighted/Filtered**: Amber (#f59e0b)
- **Dimmed**: 20% opacity

## 🔄 Switching Visualization Modes

Edit `src/index.tsx` to switch between modes:

**Columnar View** (current default):
```typescript
import AppColumnar from './AppColumnar';
render(() => <AppColumnar />, root!);
```

**Traditional Graph View**:
```typescript
import App from './App';
render(() => <App />, root!);
```

## 🔬 Pattern Mining

The columnar view includes an integrated pattern mining interface:

1. Click the **Mining Panel** button (top-right)
2. Set the **conjunct size** (number of properties in pattern)
3. Click **Mine** to discover frequent patterns
4. View results in the collapsible results card
5. Click **Visualize** on any pattern to highlight it

Mining API endpoint: `http://localhost:5000/api/mine`

## 🛠️ Development

### Project Structure
```
experiments/atomspace_visualizer/
├── src/
│   ├── components/        # React/Solid components
│   │   ├── ColumnarVisualizer/
│   │   ├── GraphVisualizer/
│   │   ├── Legend/
│   │   ├── UIControls/
│   │   └── MiningPanel/
│   ├── services/          # Business logic
│   │   ├── graph/
│   │   └── parser/
│   ├── types/             # TypeScript definitions
│   ├── utils/             # Helper functions
│   └── styles/            # Global styles
├── public/                # Static assets
└── README.md
```

### Tech Stack
- **SolidJS**: Reactive UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool and dev server
- **Canvas API**: High-performance rendering

### Building for Production
```bash
npm run build
```

Output goes to `dist/` directory.

## 📝 Technical Details

### Columnar Visualization

**ColumnarTransformer**: Converts property-based triples into columnar layout
- First column: all article IDs
- Subsequent columns: unique property types
- Rows: unique property values
- Special "None" nodes for missing values

**ColumnarVisualizer**: Canvas-based rendering with continuous curved lines
- Each article has ONE smooth line through all its properties
- Click nodes to filter and highlight
- Multi-select with Ctrl/Cmd
- Scrollable viewport for large datasets

### Traditional Graph View

**Layout Algorithms**:
- **Force-Directed**: Fruchterman-Reingold with repulsive/attractive forces
- **Hierarchical**: BFS-based level assignment with root detection
- **Circular**: Concentric circles with type-based grouping

**GraphEngine**: Smooth animations with ease-out cubic interpolation
**GraphVisualizer**: Interactive canvas with pan, zoom, and drag

## 📄 License

See the main repository LICENSE file.

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) in the main repository.

## API Reference

### Core Interfaces

```typescript
interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: GraphMetadata;
  hypergraphs: HypergraphStructure[];
}

interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  position: Point;
  color?: string;
  size?: number;
  isHypergraph?: boolean;
  metadata: {
    originalExpression?: string;
    occurrences?: number;
    isGenerated?: boolean;
  };
}

interface LayoutOptions {
  iterations?: number;
  springLength?: number;
  springStrength?: number;
  repulsionStrength?: number;
  damping?: number;
  animationDuration?: number;
  centerForce?: number;
}
```

### Key Service Methods

```typescript
// MettaParserImpl
parse(mettaText: string): ParseResult
validateSyntax(mettaText: string): ValidationResult
extractTriples(mettaText: string): Triple[]

// GraphTransformerImpl
transformTriplestoGraph(triples: Triple[]): GraphData
processSimpleTriple(triple: Triple, nodes: GraphNode[], edges: GraphEdge[], index: number): void
processHypergraphTriple(triple: Triple, ...): void

// GraphEngineImpl
setData(nodes: GraphNode[], edges: GraphEdge[]): void
applyLayout(algorithm: LayoutAlgorithm, options?: LayoutOptions): void
render(ctx: CanvasRenderingContext2D, transform: Transform): void
getLayoutState(): LayoutState
```
