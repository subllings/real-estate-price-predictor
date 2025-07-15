import React, { useState } from "react";
import axios from "axios";
import { CHAT_API_URL } from "../../config/api";
import "./ChatWidget.css";

const ChatWidget = () => {
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
        { from: "agent", text: "Sorry, I couldn't reach the assistant." }
      ]);
    }
  };

  return (
    <div className="chat-widget">
      <div className="chat-widget-header">
        <h4>💬 AI Chat Assistant</h4>
        <button 
          onClick={clearChatHistory} 
          className="clear-chat-btn"
          title="Clear chat history"
        >
          Clear
        </button>
      </div>
      
      <div className="chat-widget-messages">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`message ${msg.from === "user" ? "user-msg" : "agent-msg"}`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <div className="chat-widget-input">
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
  );
};

export default ChatWidget;
