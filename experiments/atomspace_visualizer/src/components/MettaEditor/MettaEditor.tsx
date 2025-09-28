import { Component, onMount, createSignal, createEffect, For } from 'solid-js';
import { ParseError } from '../../types';
import { MettaParserImpl } from '../../services/parser/MettaParser';
import * as shiki from 'shiki';
import mettaGrammar from '../../syntax/metta.tmLanguage.json';

export interface MettaEditorProps {
  initialText: string;
  onTextChange: (text: string) => void;
  onFileUpload: (file: File) => void;
  parseErrors: ParseError[];
}

const MettaEditor: Component<MettaEditorProps> = (props) => {
  const [text, setText] = createSignal(props.initialText);
  const [highlighted, setHighlighted] = createSignal('');
  const [realTimeErrors, setRealTimeErrors] = createSignal<ParseError[]>([]);
  const [lineNumbers, setLineNumbers] = createSignal<number[]>([]);
  let highlighter: shiki.Highlighter | undefined;
  let textareaRef: HTMLTextAreaElement | undefined;
  let highlightedRef: HTMLDivElement | undefined;
  const parser = new MettaParserImpl();

  // Build a Shiki LanguageRegistration from the JSON grammar
  const mettaLanguage: shiki.LanguageRegistration = {
    name: 'metta',
    ...mettaGrammar,
  };

  onMount(async () => {
    highlighter = await shiki.createHighlighter({
      themes: ['github-light'],
      langs: [mettaLanguage],
    });
    updateHighlighting(text());
    updateLineNumbers(text());
  });

  // Update line numbers when text changes
  const updateLineNumbers = (textValue: string) => {
    const lines = textValue.split('\n');
    setLineNumbers(lines.map((_, index) => index + 1));
  };

  // Enhanced highlighting with error overlay
  const updateHighlighting = async (textValue: string) => {
    if (!highlighter) return;

    // Get syntax highlighting with custom CSS
    let highlightedHtml = highlighter.codeToHtml(textValue, {
      lang: 'metta',
      theme: 'github-light',
      transformers: [
        {
          pre(node) {
            // Remove default pre styling to match our textarea
            node.properties.style = 'margin: 0; padding: 0; background: transparent; font-family: inherit; font-size: inherit; line-height: inherit; white-space: pre; word-wrap: break-word;';
          },
          code(node) {
            // Remove default code styling
            node.properties.style = 'font-family: inherit; font-size: inherit; line-height: inherit;';
          }
        }
      ]
    });

    // Perform real-time validation
    const validation = parser.validateSyntax(textValue);
    setRealTimeErrors([...validation.errors, ...validation.warnings]);

    // Add error highlighting to the HTML
    if (validation.errors.length > 0 || validation.warnings.length > 0) {
      highlightedHtml = addErrorHighlighting(highlightedHtml, textValue, [...validation.errors, ...validation.warnings]);
    }

    setHighlighted(highlightedHtml);
  };

  // Add error highlighting to the syntax-highlighted HTML
  const addErrorHighlighting = (html: string, textValue: string, errors: ParseError[]): string => {
    let modifiedHtml = html;

    // Create a map of line numbers to errors
    const errorsByLine = new Map<number, ParseError[]>();
    errors.forEach(error => {
      if (!errorsByLine.has(error.line)) {
        errorsByLine.set(error.line, []);
      }
      errorsByLine.get(error.line)!.push(error);
    });

    // Add error classes to lines with errors
    errorsByLine.forEach((lineErrors, lineNumber) => {
      const hasError = lineErrors.some(e => e.severity === 'error');
      const hasWarning = lineErrors.some(e => e.severity === 'warning');

      const errorClass = hasError ? 'error-line' : (hasWarning ? 'warning-line' : '');

      if (errorClass) {
        // Find the line in the HTML and add error styling
        const linePattern = new RegExp(`(<span class="line">)(.*?)(</span>)`, 'g');
        let lineIndex = 0;
        modifiedHtml = modifiedHtml.replace(linePattern, (match, openTag, content, closeTag) => {
          lineIndex++;
          if (lineIndex === lineNumber) {
            return `${openTag}<span class="${errorClass}">${content}</span>${closeTag}`;
          }
          return match;
        });
      }
    });

    return modifiedHtml;
  };

  // Handle text input with real-time validation
  const handleInput = async (e: Event) => {
    const value = (e.target as HTMLTextAreaElement).value;
    setText(value);
    props.onTextChange(value);
    updateLineNumbers(value);
    await updateHighlighting(value);
  };

  // Sync scroll between textarea and highlighted div
  const handleScroll = (e: Event) => {
    const target = e.target as HTMLTextAreaElement;
    if (highlightedRef) {
      // Use requestAnimationFrame for smooth synchronization
      requestAnimationFrame(() => {
        if (highlightedRef) {
          highlightedRef.scrollTop = target.scrollTop;
          highlightedRef.scrollLeft = target.scrollLeft;
        }
      });
    }
  };

  // Update highlighting when props.parseErrors change
  createEffect(() => {
    updateHighlighting(text());
  });

  return (
    <div style="
      display: flex; 
      flex-direction: column; 
      height: 100%; 
      width: 100%;
      box-sizing: border-box;
    ">
      <h3 style="
        margin: 0 0 12px 0; 
        font-size: 14px; 
        font-weight: 600; 
        flex-shrink: 0;
        line-height: 1.2;
      ">
        {realTimeErrors().length > 0 && (
          <span style="
            margin-left: 8px;
            font-size: 11px;
            font-weight: normal;
            color: #dc2626;
          ">
            ({realTimeErrors().filter(e => e.severity === 'error').length} errors, {realTimeErrors().filter(e => e.severity === 'warning').length} warnings)
          </span>
        )}
      </h3>

      {/* Editor Container with Line Numbers */}
      <div style="
        position: relative;
        flex: 1;
        min-height: 0;
        margin-bottom: 8px;
        border: 1px solid var(--border-light);
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.8);
        overflow: hidden;
      ">
        {/* Line Numbers */}
        <div style="
          position: absolute;
          left: 0;
          top: 0;
          width: 40px;
          height: 100%;
          background: rgba(248, 250, 252, 0.9);
          border-right: 1px solid var(--border-light);
          font-family: 'Courier New', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
          font-size: 13px;
          font-weight: normal;
          line-height: 1.5;
          padding: 8px 4px;
          margin: 0;
          box-sizing: border-box;
          overflow: hidden;
          z-index: 2;
          white-space: pre;
        ">
          <For each={lineNumbers()}>
            {(lineNum) => {
              const hasError = realTimeErrors().some(e => e.line === lineNum && e.severity === 'error');
              const hasWarning = realTimeErrors().some(e => e.line === lineNum && e.severity === 'warning');
              return (
                <div style={`
                  text-align: right;
                  color: ${hasError ? '#dc2626' : hasWarning ? '#f59e0b' : '#6b7280'};
                  font-weight: ${hasError || hasWarning ? '600' : 'normal'};
                  position: relative;
                `}>
                  {lineNum}
                  {(hasError || hasWarning) && (
                    <span style={`
                      position: absolute;
                      right: -8px;
                      top: 0;
                      width: 4px;
                      height: 100%;
                      background: ${hasError ? '#dc2626' : '#f59e0b'};
                      border-radius: 2px;
                    `}></span>
                  )}
                </div>
              );
            }}
          </For>
        </div>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text()}
          onInput={handleInput}
          onScroll={handleScroll}
          style="
            position: absolute;
            left: 44px;
            top: 0;
            right: 0;
            bottom: 0;
            font-family: 'Courier New', Consolas, 'Liberation Mono', Menlo, Courier, monospace; 
            font-size: 13px; 
            font-weight: normal;
            background: transparent;
            border: none;
            padding: 8px;
            margin: 0;
            overflow: auto;
            resize: none;
            outline: none;
            color: transparent;
            caret-color: #374151;
            z-index: 3;
            line-height: 1.5;
            white-space: pre;
            word-wrap: break-word;
            tab-size: 2;
          "
          spellcheck={false}
        />

        {/* Syntax Highlighted Overlay */}
        <div
          ref={highlightedRef}
          style="
            position: absolute;
            left: 44px;
            top: 0;
            right: 0;
            bottom: 0;
            font-family: 'Courier New', Consolas, 'Liberation Mono', Menlo, Courier, monospace; 
            font-size: 13px; 
            font-weight: normal;
            padding: 8px;
            margin: 0;
            overflow: auto;
            pointer-events: none;
            z-index: 1;
            line-height: 1.5;
            white-space: pre;
            word-wrap: break-word;
            tab-size: 2;
          "
          innerHTML={highlighted()}
        />
      </div>

      {/* Action Buttons */}
      <div style="
        margin-bottom: 8px; 
        display: flex; 
        gap: 8px; 
        flex-shrink: 0;
        align-items: center;
      ">
        <button
          style="
            padding: 4px 8px; 
            font-size: 11px; 
            border: 1px solid var(--border-light); 
            border-radius: 3px; 
            background: white; 
            cursor: pointer;
            transition: all 0.2s ease;
          "
          onClick={() => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.metta,.txt';
            input.onchange = (e) => {
              const file = (e.target as HTMLInputElement).files?.[0];
              if (file) props.onFileUpload(file);
            };
            input.click();
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-primary)'}
          onMouseLeave={(e) => e.currentTarget.style.background = 'white'}
        >
          Load File
        </button>
        <button
          style="
            padding: 4px 8px; 
            font-size: 11px; 
            border: 1px solid var(--border-light); 
            border-radius: 3px; 
            background: white; 
            cursor: pointer;
            transition: all 0.2s ease;
          "
          onClick={() => {
            setText('');
            props.onTextChange('');
            updateLineNumbers('');
            updateHighlighting('');
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-primary)'}
          onMouseLeave={(e) => e.currentTarget.style.background = 'white'}
        >
          Clear
        </button>
      </div>

      {/* Error Display Panel */}
      {(realTimeErrors().length > 0 || props.parseErrors.length > 0) && (
        <div style="
          max-height: 120px;
          overflow-y: auto;
          border: 1px solid var(--border-light);
          border-radius: 4px;
          background: rgba(255, 255, 255, 0.95);
          flex-shrink: 0;
        ">
          <div style="
            padding: 8px;
            font-size: 11px;
            font-weight: 600;
            background: rgba(248, 250, 252, 0.8);
            border-bottom: 1px solid var(--border-light);
          ">
            Issues ({realTimeErrors().filter(e => e.severity === 'error').length + props.parseErrors.filter(e => e.severity === 'error').length} errors, {realTimeErrors().filter(e => e.severity === 'warning').length + props.parseErrors.filter(e => e.severity === 'warning').length} warnings)
          </div>
          <div style="padding: 4px;">
            <For each={[...realTimeErrors(), ...props.parseErrors]}>
              {(error) => (
                <div style={`
                  padding: 4px 8px;
                  margin: 2px 0;
                  border-left: 3px solid ${error.severity === 'error' ? '#dc2626' : '#f59e0b'};
                  background: ${error.severity === 'error' ? 'rgba(220, 38, 38, 0.05)' : 'rgba(245, 158, 11, 0.05)'};
                  border-radius: 2px;
                  font-size: 11px;
                  line-height: 1.3;
                `}>
                  <div style={`
                    font-weight: 600;
                    color: ${error.severity === 'error' ? '#dc2626' : '#f59e0b'};
                    margin-bottom: 2px;
                  `}>
                    {error.severity === 'error' ? '⚠️' : '⚡'} Line {error.line}:{error.column}
                  </div>
                  <div style="color: #374151;">
                    {error.message}
                  </div>
                </div>
              )}
            </For>
          </div>
        </div>
      )}
    </div>
  );
};

export default MettaEditor;