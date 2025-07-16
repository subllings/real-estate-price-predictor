import React, { useState } from "react";
import "./SidePanel.css";
import axios from "axios";
import { CHAT_API_URL } from "../../config/api";

const SidePanel = ({ user, isExpanded, onToggle, onClose, comments }) => {
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState([
    { from: "agent", text: "Hello! How can I assist you today?" }
  ]);

  const clearChatHistory = () => {
    setMessages([
      { from: "agent", text: "Hello! How can I assist you today?" }
    ]);
  };

const handleSend = async () => {
  if (!chatInput.trim()) return;

  const userMessage = { role: "user", content: chatInput };

  // Ajout côté UI (affichage)
  setMessages(prev => [...prev, { from: "user", text: chatInput }]);
  setChatInput("");

  try {
    const response = await axios.post(CHAT_API_URL, {
      messages: [userMessage]  
    });

    // La réponse attendue dans response.data.response
    setMessages(prev => [
      ...prev,
      { from: "agent", text: response.data.response || "No response from assistant." }
    ]);
  } catch (err) {
    console.error("Chat error:", err.response?.data || err.message || err);
    setMessages(prev => [
      ...prev,
      { from: "agent", text: "Sorry, I couldn’t reach the assistant." }
    ]);
  }
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
            <button className="close-btn" onClick={onClose} aria-label="Close Side Panel">
              &times;
            </button>
          </>
        )}
      </header>

      {isExpanded && (
        <div className="sidepanel-content">
          
          <section className="llm-comment">
            <h4>AI Commentary</h4>
            {comments.length === 0 ? (
              <p>No comment available.</p>
            ) : (
              <div className="comments-list">
                {comments.map((comment, idx) => (
                  <p key={idx} className={comment.startsWith('===') ? 'prediction-header' : 'comment-text'}>
                    {comment}
                  </p>
                ))}
              </div>
            )}
          </section>

          <section className="chat-section">
            <div className="chat-header">
              <h4>Chat</h4>
              <button 
                onClick={clearChatHistory} 
                className="clear-chat-btn"
                title="Clear chat history"
              >
                Clear
              </button>
            </div>
            
            <div className="chat-messages">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`message ${msg.from === "user" ? "user-msg" : "agent-msg"}`}
                >
                  {msg.text}
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {isExpanded && (
        <div className="chat-input-container">
          <div className="chat-input-area">
            <input
              type="text"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder="Ask your question..."
              onKeyDown={e => {
                if (e.key === "Enter") handleSend();
              }}
              aria-label="Chat input"
            />
            <button onClick={handleSend} className="send-btn" aria-label="Send message">
              Send
            </button>
          </div>
        </div>
      )}
    </aside>
  );
};

export default SidePanel;
