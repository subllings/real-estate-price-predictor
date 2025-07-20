import React from 'react';

// Helper function to render inline formatting (bold, etc.)
const renderInlineFormatting = (text) => {
  if (!text) return text;

  // Handle **bold** text
  const parts = [];
  const boldRegex = /\*\*(.*?)\*\*/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = boldRegex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    
    // Add the bold text
    parts.push(
      <strong key={key++} className="font-bold text-blue-600">
        {match[1]}
      </strong>
    );
    
    lastIndex = boldRegex.lastIndex;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }
  
  // If no formatting found, return original text
  return parts.length > 0 ? parts : text;
};

// Main markdown renderer
const renderMarkdownContent = (content) => {
  console.log('🔍 renderMarkdownContent called with content:', content?.substring(0, 100) + '...');
  
  if (!content) {
    console.log('❌ No content provided');
    return null;
  }

  const lines = content.split('\n');
  const elements = [];
  let i = 0;

  console.log('📝 Processing', lines.length, 'lines of content');

  while (i < lines.length) {
    const line = lines[i].trim();

    // Horizontal rule
    if (line === '---') {
      console.log('➖ Found horizontal rule');
      elements.push(
        <hr key={elements.length} className="my-4 border-t border-gray-300" />
      );
      i++;
      continue;
    }

    // Blockquote
    if (line.startsWith('>')) {
      console.log('💬 Found blockquote');
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith('>')) {
        quoteLines.push(lines[i].replace(/^>\s*/, ''));
        i++;
      }
      elements.push(
        <blockquote key={elements.length} className="border-l-4 border-blue-300 pl-4 italic text-gray-700 my-4">
          {quoteLines.map((q, idx) => <p key={idx}>{renderInlineFormatting(q)}</p>)}
        </blockquote>
      );
      continue;
    }

    // Checkboxes
    if (line.startsWith('- [ ]') || line.startsWith('- [x]')) {
      console.log('☑️ Found checkboxes');
      const items = [];
      while (i < lines.length && (lines[i].startsWith('- [ ]') || lines[i].startsWith('- [x]'))) {
        const isChecked = lines[i].startsWith('- [x]');
        const label = lines[i].slice(5).trim();
        items.push({ label, isChecked });
        i++;
      }

      elements.push(
        <ul key={elements.length} className="pl-6 mt-2 mb-2">
          {items.map((item, idx) => (
            <li key={idx} className="mb-1 flex items-center space-x-2">
              <input type="checkbox" checked={item.isChecked} readOnly className="form-checkbox" />
              <span>{renderInlineFormatting(item.label)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Markdown table
    if (line.startsWith('|') && line.endsWith('|')) {
      console.log('📊 Found table');
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const currentLine = lines[i].trim();
        if (!currentLine.includes('---')) {
          tableLines.push(currentLine);
        }
        i++;
      }

      const tableRows = tableLines.map(l =>
        l.split('|').slice(1, -1).map(cell => cell.trim())
      );

      const headers = tableRows[0];
      const rows = tableRows.slice(1);

      elements.push(
        <table key={elements.length} className="w-full border-collapse border border-gray-300 mt-4 mb-4">
          <thead className="bg-gray-100">
            <tr>
              {headers.map((cell, idx) => (
                <th key={idx} className="border border-gray-300 px-4 py-2 text-left font-semibold">
                  {renderInlineFormatting(cell)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr key={rowIdx}>
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx} className="border border-gray-300 px-4 py-2">
                    {renderInlineFormatting(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
      continue;
    }

    // Headers
    if (line.startsWith('####')) {
      console.log('📝 Found h4 header:', line);
      elements.push(
        <h4 key={elements.length} className="text-lg font-bold text-blue-600 mt-4 mb-2">
          {renderInlineFormatting(line.replace(/^####\s*/, ''))}
        </h4>
      );
      i++;
      continue;
    }

    if (line.startsWith('###')) {
      console.log('📝 Found h3 header:', line);
      elements.push(
        <h3 key={elements.length} className="text-xl font-bold text-blue-600 mt-4 mb-2">
          {renderInlineFormatting(line.replace(/^###\s*/, ''))}
        </h3>
      );
      i++;
      continue;
    }

    // Unordered list
    if (line.startsWith('- ')) {
      console.log('📋 Found unordered list');
      const items = [];
      while (i < lines.length && lines[i].trim().startsWith('- ')) {
        items.push(lines[i].trim().replace(/^- /, ''));
        i++;
      }
      elements.push(
        <ul key={elements.length} className="list-disc pl-6 mt-2 mb-2">
          {items.map((item, idx) => (
            <li key={idx}>{renderInlineFormatting(item)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\d+\.\s/.test(line)) {
      console.log('🔢 Found ordered list');
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s*/, ''));
        i++;
      }
      elements.push(
        <ol key={elements.length} className="list-decimal pl-6 mt-2 mb-2">
          {items.map((item, idx) => (
            <li key={idx}>{renderInlineFormatting(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // Paragraph
    if (line.length > 0) {
      console.log('📄 Found paragraph:', line.substring(0, 50) + '...');
      elements.push(
        <p key={elements.length} className="mb-2">
          {renderInlineFormatting(line)}
        </p>
      );
    }

    i++;
  }

  console.log('✅ Rendered', elements.length, 'elements');
  return <div>{elements}</div>;
};

export default renderMarkdownContent;
