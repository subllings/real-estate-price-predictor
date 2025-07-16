import React, { useState, useEffect, useRef } from "react";
import "./SidePanel.css";
import axios from "axios";
import { CHAT_API_URL } from "../../config/api";

const SidePanel = ({ user, isExpanded, onToggle, onClose, comments, clearComments }) => {
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState([
    { from: "agent", text: "Hello! How can I assist you today?" }
  ]);

  // Référence pour le scroll automatique sur tout le side panel
  const sidePanelRef = useRef(null);

  // Ajouter les commentaires de prédiction comme messages dans le chat
  useEffect(() => {
    if (comments && comments.length > 0) {
      const newComments = comments.map(comment => {
        // Vérifier si c'est un titre de prédiction (contient " in " et se termine par " - HH:MM:SS AM/PM")
        const isPredictionTitle = comment.includes(' in ') && /\s-\s\d{1,2}:\d{2}:\d{2}\s(AM|PM)$/.test(comment);
        // Vérifier si c'est un titre d'analyse ESG
        const isESGTitle = comment.startsWith('ESG Analysis for') && /\s-\s\d{1,2}:\d{2}:\d{2}\s(AM|PM)$/.test(comment);

        let subtype = "prediction-comment";
        if (isPredictionTitle) {
          subtype = "prediction-title";
        } else if (isESGTitle) {
          subtype = "esg-title";
        }

        return {
          from: "agent",
          text: comment,
          type: "prediction",
          subtype: subtype
        };
      });

      setMessages(prev => {
        // Éviter les doublons en filtrant les messages de prédiction existants
        const withoutPredictions = prev.filter(msg => msg.type !== "prediction");
        return [...withoutPredictions, ...newComments];
      });
    }
  }, [comments]);

  // Scroll automatique vers le bas quand de nouveaux messages sont ajoutés
  useEffect(() => {
    if (sidePanelRef.current && isExpanded) {
      setTimeout(() => {
        sidePanelRef.current.scrollTop = sidePanelRef.current.scrollHeight;
      }, 100); // Petit délai pour s'assurer que le contenu est rendu
    }
  }, [messages, isExpanded]);

  const clearChatHistory = () => {
    setMessages([
      { from: "agent", text: "Hello! How can I assist you today?" }
    ]);
    // Aussi effacer les commentaires si la fonction est fournie
    if (clearComments) {
      clearComments();
    }
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
    <>
      {/* Onglet visible pour rouvrir le panel */}
      <div
        className={`sidepanel-tab ${isExpanded ? 'hidden' : ''}`}
        onClick={onToggle}
        title="Open AI Chat Assistant"
      >
        <div className="sidepanel-tab-icon">CHAT</div>
      </div>

      <aside className={`sidepanel ${isExpanded ? "open" : ""}`}>
        {isExpanded && (
          <>
            <div className="sidepanel-header">
              <h3>Profile: {user.profile}</h3>
              <button className="close-btn" onClick={onClose} aria-label="Close Side Panel">
                &times;
              </button>
            </div>

            <div ref={sidePanelRef} className="sidepanel-content">
              <section className="chat-section">
                <div className="chat-header">
                  <h4>AI Chat Assistant</h4>
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
                      className={`message ${msg.from === "user" ? "user-msg" : "agent-msg"} ${msg.type === "prediction" ? "prediction-msg" : ""} ${msg.subtype === "prediction-title" ? "prediction-title" 
                        : ""} ${msg.subtype === "esg-title" ? "esg-title" : ""} ${msg.subtype === "prediction-comment" ? "prediction-comment" : ""}`}
                    >
                      {msg.text}
                    </div>
                  ))}
                </div>
              </section>
            </div>

            <div className="chat-input">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask your question..."
              />
              <button onClick={handleSend}>Send</button>
            </div>
          </>
        )}
      </aside>
    </>
  );
};

export default SidePanel;
