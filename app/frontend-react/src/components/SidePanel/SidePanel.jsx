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
          subtype: subtype,
          timestamp: new Date().toISOString()
        };
      });

      setMessages(prev => {
        // Garder tous les messages existants et ajouter les nouveaux
        // Éviter les doublons en filtrant seulement les messages de prédiction identiques
        const existingTexts = prev.filter(msg => msg.type === "prediction").map(msg => msg.text);
        const uniqueNewComments = newComments.filter(newComment => 
          !existingTexts.includes(newComment.text)
        );
        
        return [...prev, ...uniqueNewComments];
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
    setMessages(prev => [...prev, { from: "user", text: chatInput, timestamp: new Date().toISOString() }]);
    setChatInput("");

    try {
      // Préparer l'historique des conversations (derniers 20 messages)
      const conversationHistory = messages.slice(-20).map(msg => ({
        role: msg.from === "user" ? "user" : "assistant",
        content: msg.text
      }));

      // Ajouter le message actuel
      conversationHistory.push(userMessage);

      // Ajouter un message système avec contexte pour l'IA
      const messagesWithContext = [
        {
          role: "system",
          content: "You are a helpful real estate AI assistant. You have access to conversation history and can provide contextual responses based on previous property predictions and discussions. Keep responses concise, helpful, and professional. You can reference earlier predictions and continue conversations naturally."
        },
        ...conversationHistory
      ];

      const response = await axios.post(CHAT_API_URL, {
        messages: messagesWithContext
      });

      // La réponse attendue dans response.data.response
      setMessages(prev => [
        ...prev,
        { from: "agent", text: response.data.response || "No response from assistant.", timestamp: new Date().toISOString() }
      ]);
    } catch (err) {
      console.error("Chat error:", err.response?.data || err.message || err);
      setMessages(prev => [
        ...prev,
        { from: "agent", text: "Sorry, I couldn't reach the assistant.", timestamp: new Date().toISOString() }
      ]);
    }
  };

  // Fonction pour convertir le markdown simple (**texte**) en HTML
  const formatMessageText = (text) => {
    if (!text) return { __html: '' };
    
    let formattedText = text.toString();
    
    // 1. D'ABORD convertir **texte** en <strong>texte</strong> (plus restrictif)
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 2. Convertir les séparateurs markdown en lignes de séparation
    formattedText = formattedText.replace(/---+/g, '<hr style="border: none; border-top: 1px solid #ddd; margin: 10px 0;"/>');
    
    // 3. Convertir les titres markdown # ## ### en titres HTML (SUPPRIMER complètement les #)
    formattedText = formattedText.replace(/^#{1}\s+(.*?)$/gm, '<h3 style="margin: 15px 0 10px 0; font-weight: bold; color: #2563eb; font-size: 18px;">$1</h3>');
    formattedText = formattedText.replace(/^#{2}\s+(.*?)$/gm, '<h4 style="margin: 12px 0 8px 0; font-weight: bold; color: #333; font-size: 16px;">$1</h4>');
    formattedText = formattedText.replace(/^#{3}\s+(.*?)$/gm, '<h5 style="margin: 8px 0 6px 0; font-weight: bold; color: #444; font-size: 14px;">$1</h5>');
    
    // 4. Convertir les puces • + - en listes HTML compactes (AUCUN margin)
    formattedText = formattedText.replace(/^[•\+\-]\s*(.*?)$/gm, '<div style="margin: 0; padding: 0 0 0 15px; line-height: 1.3;">• $1</div>');
    
    // 5. Nettoyer les sauts de ligne AVANT de les convertir
    formattedText = formattedText.replace(/\n\s*\n\s*\n/g, '\n\n'); // Supprimer les triple+ sauts de ligne
    
    // 6. Convertir les sauts de ligne simples en <br/> mais éviter autour des balises HTML
    formattedText = formattedText.replace(/\n(?!\s*<)/g, '<br/>');
    
    // 7. Nettoyer les <br/> en trop autour des éléments HTML
    formattedText = formattedText.replace(/(<br\/>){3,}/g, '<br/><br/>');
    formattedText = formattedText.replace(/<br\/>\s*(<h[1-6])/g, '$1');
    formattedText = formattedText.replace(/(<\/h[1-6]>)\s*<br\/>/g, '$1');
    formattedText = formattedText.replace(/<br\/>\s*(<hr)/g, '$1');
    formattedText = formattedText.replace(/(<\/hr>)\s*<br\/>/g, '$1');
    formattedText = formattedText.replace(/<br\/>\s*(<div)/g, '$1');
    formattedText = formattedText.replace(/(<\/div>)\s*<br\/>/g, '$1');
    
    return { __html: formattedText };
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
                      dangerouslySetInnerHTML={formatMessageText(msg.text)}
                    />
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
