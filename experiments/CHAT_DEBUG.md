# Chat Feature Debug Guide

## Expected Behavior

1. **Open browser**: http://localhost:3000
2. **Click "Mine Neural Gold" button** (set conjunct count to 2 or 3 first)
3. **Chat should appear** from bottom, below the mine button
4. **Auto-message**: Chat should show "Mine rules with {N} patterns" as if you typed it
5. **AI responds**: AI should respond to your mining request
6. **Continue chatting**: You can ask anything

## What to Check in Browser Console (F12)

Look for these console logs:

```
ChatInterface rendered, props: {conjunctSize: undefined, isChatOpen: false}
ChatInterface: conjunctSize changed to: undefined last: null isChatOpen: false

[After clicking Mine button:]
ChatInterface: conjunctSize changed to: 2 last: null isChatOpen: false
ChatInterface: Opening chat and sending message for conjunct size: 2
```

## If Chat Doesn't Appear

### Check 1: Is ChatInterface rendering?
Look for: `ChatInterface rendered` in console

- ✅ **YES**: Component is loading
- ❌ **NO**: Check if ChatInterface is imported in App.tsx

### Check 2: Is conjunctSize changing?
Look for: `conjunctSize changed to: 2`

- ✅ **YES**: Prop is being passed correctly
- ❌ **NO**: MiningInterface isn't calling handlePatternsFound with conjunctSize

### Check 3: Is isChatOpen becoming true?
Look for: `Opening chat and sending message`

- ✅ **YES**: Logic is firing, CSS issue
- ❌ **NO**: Effect isn't triggering

### Check 4: Can you see it in DOM?
1. Open DevTools (F12)
2. Go to Elements tab
3. Search for `chat-interface` class
4. Check if element exists and has `display: flex`

## CSS Positioning

Chat should be:
- **Position**: Fixed at bottom
- **Width**: 500px
- **Height**: calc(100vh - 180px) - full height minus mining button area
- **Z-index**: 9999 (very high to be on top)
- **Location**: Centered horizontally

## Manual Test

If auto-trigger doesn't work, you can manually trigger it:

1. Open browser console (F12)
2. After page loads, type:
```javascript
// This should force the chat open
document.querySelector('.chat-toggle-btn')?.click()
```

## Current Implementation

**File**: `ChatInterface.tsx`
```typescript
// Effect watches for conjunctSize changes
createEffect(() => {
  const conjunctSize = props.conjunctSize;
  if (conjunctSize && conjunctSize !== lastConjunctSize() && conjunctSize > 0) {
    setIsChatOpen(true);  // Opens chat
    // Auto-sends message
    const userMessage = `Mine rules with ${conjunctSize} patterns`;
    // ... sends to AI
  }
});
```

**File**: `App.tsx`
```typescript
// Passes conjunctSize to ChatInterface
<ChatInterface 
  onVisualize={handleVisualize}
  miningResults={miningResults()}
  conjunctSize={currentConjunctSize()}  // <-- This should change when mining starts
/>
```

**File**: `MiningInterface.tsx`
```typescript
// Calls parent with conjunctSize
props.onPatternsFound?.(jobData.result, conjunctionCount());
```

## Quick Fixes

### Fix 1: Force Chat Open
If everything else works but chat doesn't open, try setting initial state to true:

```typescript
const [isChatOpen, setIsChatOpen] = createSignal(true); // Force open for testing
```

### Fix 2: Check Mining Button Position
The chat appears below mining button. Make sure mining button is visible and not hidden.

### Fix 3: Check for CSS Conflicts
Search for other elements with high z-index that might cover the chat.

## Test Sequence

1. ✅ Open http://localhost:3000
2. ✅ Open browser console (F12)
3. ✅ Set conjunct count to 2
4. ✅ Click "Mine Neural Gold"
5. ✅ Watch console for logs
6. ✅ Look for chat appearing from bottom
7. ✅ Verify message "Mine rules with 2 patterns" appears
8. ✅ Verify AI responds

## Success Criteria

✅ Chat appears smoothly from bottom with slide animation  
✅ Message "Mine rules with {N} patterns" appears as user message  
✅ AI responds with mining acknowledgment  
✅ You can continue typing and chatting  
✅ Chat stays open until you close it  

## Still Not Working?

Share these in your message:
1. Screenshot of browser console logs
2. Screenshot of Elements tab showing chat-interface
3. Any error messages in console
4. Current conjunct count value when clicking Mine
