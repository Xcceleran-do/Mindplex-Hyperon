# Quick Start Guide - Columnar Visualization

## Running the Application

```bash
cd /workspaces/Mindplex-Hyperon/experiments/atomspace_visualizer
npm install
npm run dev
```

Visit: http://localhost:3000/

## Visual Guide

### What You'll See

```
┌─────────────┬──────────┬──────────┬──────────────┬─────────────────────┐
│  ARTICLES   │  TOPIC   │  LENGTH  │    TONE      │   WRITING_STYLE     │
├─────────────┼──────────┼──────────┼──────────────┼─────────────────────┤
│ Article 0 ──┼─→ AI     │          │              │                     │
│             │          ├─→ high   │              │                     │
│ Article 1   │          │          ├─→ Analytical │                     │
│             │          │          │              ├─→ Formal            │
│ Article 2   │          │          │              │                     │
│             │          ├─→ low    │              ├─→ Conversational    │
│ Article 3   │          │          │              │                     │
│             │          ├─→ medium │              ├─→ Narrative         │
│ ...         │          │          │              │                     │
│             │          │          ├─→ Humorous   │                     │
└─────────────┴──────────┴──────────┴──────────────┴─────────────────────┘
```

**Note**: Each article has ONE continuous curved line flowing through all its property values!

## Key Features

### 1. Continuous Line Path
- Each article has a **single smooth line** connecting all its properties
- The line **curves** at each property value
- Easy to trace an article's complete profile

### 2. Multi-Select Filtering

#### Single Selection
```
Click → Article 0
Result: Only Article 0 and its properties are highlighted
```

#### Multi-Selection
```
Click → Article 0
Ctrl+Click → Article 1
Ctrl+Click → Article 2
Result: All three articles and their properties are highlighted
```

#### Mixed Filtering
```
Click → Article 0
Ctrl+Click → "Humorous" (in TONE column)
Result: Article 0 AND all articles with "Humorous" tone are highlighted
```

### 3. Interactive Legend

**Collapsed by Default** → Click header to expand

Inside the legend:
- **Articles**: Click to filter by article
- **Properties**: Click to filter by property value
- **Multi-select**: Hold Ctrl/Cmd while clicking
- **Clear All**: Reset all filters

### 4. Navigation

| Action | How |
|--------|-----|
| Pan | Click + Drag empty space |
| Zoom In | Scroll up |
| Zoom Out | Scroll down |
| Reset View | Click reset button in controls |

## Understanding the Data

### The Sample Data (small-ugly.metta)
- **10 articles** (0-9)
- **5 property types**: topic, length, tone, writing_style, engagement_level
- Each article has different property values
- Some articles are missing certain properties (connected to "None")

### Example: Article 0
```
Article 0:
├─ topic: AI
├─ length: high
├─ tone: Analytical
├─ writing_style: Formal
└─ engagement_level: high
```

This creates ONE continuous line: 
```
Article 0 → AI → high → Analytical → Formal → high
```

## Pro Tips

1. **Hold Ctrl/Cmd** to select multiple items without clearing previous selection
2. **Click empty space** to clear all filters (if not holding Ctrl/Cmd)
3. **Use the legend** for quick filtering instead of clicking the canvas
4. **Zoom in** to see details of overlapping lines
5. **Collapse panels** to maximize visualization space

## Troubleshooting

### Lines are overlapping
- **Zoom in** for better view
- **Filter** to show only relevant articles

### Can't select multiple items
- Make sure you're **holding Ctrl** (Windows/Linux) or **Cmd** (Mac) while clicking

### Legend is not visible
- Click the **collapsed legend panel** at bottom-left
- It starts minimized to save space

### View is too zoomed in/out
- Click the **reset button** in the controls panel (top-right)
- Or use scroll wheel to adjust zoom

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl/Cmd + Click | Multi-select |
| Scroll Wheel | Zoom in/out |
| Click + Drag | Pan view |

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Articles | Blue | #3b82f6 |
| Headers | Purple | #8b5cf6 |
| Property Values | Green | #10b981 |
| Highlighted | Amber | #f59e0b |
| None (missing) | Gray | #6b7280 |
| Dimmed | Transparent | rgba(*, *, *, 0.2) |

## Next Steps

1. **Explore the data** by clicking different articles
2. **Try multi-select** to compare multiple articles
3. **Filter by properties** to find patterns
4. **Export your findings** (feature coming soon)

Enjoy exploring your data! 🎉
