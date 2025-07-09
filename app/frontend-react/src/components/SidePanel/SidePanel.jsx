import React, { useState } from "react";
import "./SidePanel.css";

const SidePanel = ({ user, isExpanded, onToggle, onClose }) => {
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState([
    { from: "agent", text: "Hello! How can I assist you today?" }
  ]);

  const handleSend = () => {
    if (!chatInput.trim()) return;

    setMessages(prev => [...prev, { from: "user", text: chatInput }]);
    const userMessage = chatInput;
    setChatInput("");

    setTimeout(() => {
      setMessages(prev => [...prev, { from: "agent", text: `You asked: "${userMessage}"` }]);
    }, 1000);
  };

  return (
    <aside className={`sidepanel ${isExpanded ? "expanded" : "collapsed"}`}>
      <header className="sidepanel-header">
        <button
          className="toggle-btn"
          onClick={onToggle}
          aria-label={isExpanded ? "Collapse side panel" : "Expand side panel"}
        >
          {isExpanded ? "«" : "»"}
        </button>
        {isExpanded && (
          <>
            <h3>Profile: {user.profile}</h3>
            <button className="close-btn" onClick={onClose} aria-label="Close Side Panel">&times;</button>
          </>
        )}
      </header>

      {isExpanded && (
        <>
          <section className="search-history">
            <h4>Search History</h4>
            <ul>
              {user.history.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </section>

          <section className="chat-container">
            <div className="messages" aria-live="polite" aria-atomic="true">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`message ${msg.from === "user" ? "user-msg" : "agent-msg"}`}
                >
                  {msg.text}
                </div>
              ))}
            </div>

            <div className="chat-input-area">
              <input
                type="text"
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                placeholder="Ask your question..."
                onKeyDown={e => { if (e.key === "Enter") handleSend(); }}
                aria-label="Chat input"
              />
              <button onClick={handleSend} className="send-btn" aria-label="Send message">
                Send
              </button>
            </div>
          </section>
        </>
      )}
    </aside>
  );
};

export default SidePanel;
