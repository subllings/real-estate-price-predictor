import React, { useState, useEffect, useRef } from "react";
import "./SidePanel.css";
import axios from "axios";
import { CHAT_API_URL } from "../../config/api";

const SidePanel = ({ user, isExpanded, onToggle, onClose, comments, clearComments, propertyData, predictionData, esgData, onSendChatMessage, onResetStrategicAnalysis }) => {
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState([
    { from: "agent", text: "Hello! How can I assist you today?" }
  ]);

  // Référence pour le scroll automatique sur tout le side panel
  const sidePanelRef = useRef(null);
  
  // Référence pour le conteneur des messages spécifiquement
  const messagesContainerRef = useRef(null);
  
  // Référence pour le panneau pour le redimensionnement
  const panelRef = useRef(null);

  // État pour suivre si l'analyse stratégique a déjà été générée
  const [strategicAnalysisGenerated, setStrategicAnalysisGenerated] = useState(false);
  
  // État pour le spinner de chargement de l'analyse stratégique
  const [isStrategicAnalysisLoading, setIsStrategicAnalysisLoading] = useState(false);

  // États pour le redimensionnement
  const [panelWidth, setPanelWidth] = useState(500);
  const [isResizing, setIsResizing] = useState(false);
  const [startX, setStartX] = useState(0);
  const [startWidth, setStartWidth] = useState(500);

  // Fonction pour réinitialiser l'analyse stratégique
  const resetStrategicAnalysis = () => {
    setStrategicAnalysisGenerated(false);
    setIsStrategicAnalysisLoading(false);
    console.log("Strategic analysis states reset");
  };

  // Exposer la fonction au composant parent
  useEffect(() => {
    if (onResetStrategicAnalysis) {
      onResetStrategicAnalysis(resetStrategicAnalysis);
    }
  }, [onResetStrategicAnalysis]);

  // Fonction pour envoyer un message au chat depuis l'extérieur
  const sendMessageToChat = async (message) => {
    if (!message || !message.trim()) return;

    // Ajouter le message à l'input
    setChatInput(message);
    
    // Ajouter le message utilisateur à la liste
    setMessages(prev => [...prev, { from: "user", text: message, timestamp: new Date().toISOString() }]);

    try {
      // Préparer l'historique des conversations (derniers 20 messages)
      const conversationHistory = messages.slice(-20).map(msg => ({
        role: msg.from === "user" ? "user" : "assistant",
        content: msg.text
      }));

      // Ajouter le message actuel
      conversationHistory.push({ role: "user", content: message });

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
      
      // Déclencher le scroll automatique après la réponse
      setTimeout(() => {
        scrollToBottomSmooth();
      }, 100);
    } catch (err) {
      console.error("Chat error:", err.response?.data || err.message || err);
      setMessages(prev => [
        ...prev,
        { from: "agent", text: "Sorry, I couldn't reach the assistant.", timestamp: new Date().toISOString() }
      ]);
      
      // Déclencher le scroll automatique même en cas d'erreur
      setTimeout(() => {
        scrollToBottomSmooth();
      }, 100);
    }
    
    // Vider l'input après l'envoi
    setChatInput("");
  };

  // Exposer la fonction via le callback
  useEffect(() => {
    if (onSendChatMessage) {
      onSendChatMessage(sendMessageToChat);
    }
  }, [onSendChatMessage, sendMessageToChat]);

  // Déclenchement automatique de l'analyse stratégique quand l'ESG analysis est terminée
  useEffect(() => {
    console.log("ESG Data:", esgData);
    console.log("Strategic Analysis Generated:", strategicAnalysisGenerated);
    console.log("Property Data:", propertyData);
    
    // Déclencher l'analyse stratégique si :
    // 1. On a des données ESG (soit esg_scores, soit d'autres données ESG)
    // 2. L'analyse stratégique n'a pas encore été générée
    // 3. On a des données de propriété
    if (esgData && !strategicAnalysisGenerated && propertyData) {
      console.log("ESG analysis completed, triggering automatic strategic analysis...");
      
      // Délai pour laisser l'ESG analysis se finaliser
      setTimeout(() => {
        generateStrategicAnalysis();
        // Ne pas définir setStrategicAnalysisGenerated(true) ici
        // car cela sera fait dans generateStrategicAnalysis() seulement en cas de succès
      }, 2000); // 2 secondes de délai
    }
  }, [esgData, strategicAnalysisGenerated, propertyData]);

  // Reset strategic analysis state when property data changes
  useEffect(() => {
    setStrategicAnalysisGenerated(false);
    setIsStrategicAnalysisLoading(false);
  }, [propertyData]);

  // Ajouter les commentaires de prédiction comme messages dans le chat
  useEffect(() => {
    if (comments && comments.length > 0) {
      // Combiner les commentaires connexes
      const combinedComments = [];
      let i = 0;
      
      while (i < comments.length) {
        const currentComment = comments[i];
        
        // Combiner les messages ESG consécutifs
        if (currentComment.startsWith('ESG Analysis for')) {
          // Utiliser le message réel envoyé depuis PropertyForm
          let combinedText = currentComment;
          
          // Ajouter les messages ESG complémentaires qui suivent
          while (i + 1 < comments.length && 
                 (comments[i + 1].includes('ESG analysis completed') || 
                  comments[i + 1].includes('right panel') ||
                  comments[i + 1].includes('Detailed analysis and scores') ||
                  comments[i + 1].trim() === '')) {
            i++; // Skip les messages complémentaires
            if (comments[i].trim() !== '') {
              combinedText += '\n' + comments[i];
            }
          }
          
          combinedComments.push(combinedText);
        } 
        // Combiner les messages de prédiction prix consécutifs
        else if (currentComment.startsWith('Predicted price:')) {
          let combinedText = currentComment;
          
          // Chercher les informations de modèle qui suivent
          if (i + 1 < comments.length && comments[i + 1].startsWith('Model:')) {
            combinedText += '\n' + comments[i + 1];
            i++; // Skip le prochain commentaire
          }
          
          combinedComments.push(combinedText);
        }
        else {
          combinedComments.push(currentComment);
        }
        
        i++;
      }

      const newComments = combinedComments.map(comment => {
        let subtype = "prediction-comment";
        
        // Debug: log le commentaire pour vérifier
        console.log("Processing comment:", comment);
        
        // Détecter les différents types de messages avec une meilleure logique
        if (comment.startsWith('ESG Analysis for')) {
          subtype = "esg-title"; // Tous les messages ESG Analysis
          console.log("ESG message detected, setting esg-title");
        } else if (comment.startsWith('Complete Analysis') || 
            comment.startsWith('Prediction for') ||
            (comment.includes(' in ') && comment.includes('('))) {
          subtype = "prediction-title";
        } else if (comment.startsWith('Predicted price:')) {
          subtype = "prediction-title"; // Traiter le prix comme un titre aussi
        } else if (comment.startsWith('Model:')) {
          subtype = "model-info"; // Nouveau type pour les informations de modèle
        } else if (comment.includes('ESG ANALYSIS') ||
                   comment.includes('ESG analysis') ||
                   comment.includes('available in right panel') ||
                   comment.includes('right panel →')) {
          subtype = "esg-title"; // Messages ESG en bleu pour cohérence
        } else if (comment.startsWith('Strategic Analysis') || 
                   comment.includes('STRATEGIC ANALYSIS')) {
          subtype = "strategic-title";
        }

        console.log("Final subtype:", subtype);

        return {
          from: "agent",
          text: comment,
          type: "prediction",
          subtype: subtype,
          timestamp: new Date().toISOString()
        };
      });

      setMessages(prev => {
        // Filtrer seulement les nouveaux messages uniques
        const existingTexts = prev.map(msg => msg.text);
        const uniqueNewComments = newComments.filter(newComment => 
          !existingTexts.includes(newComment.text)
        );
        
        // Si on a de nouveaux commentaires, les ajouter
        if (uniqueNewComments.length > 0) {
          return [...prev, ...uniqueNewComments];
        }
        
        return prev;
      });
    }
  }, [comments]);

  // Fonction utilitaire pour le défilement automatique amélioré
  const scrollToBottomSmooth = () => {
    const scrollContainer = messagesContainerRef.current || sidePanelRef.current;
    if (scrollContainer) {
      // Utiliser scrollTo avec behavior: 'smooth' pour un défilement fluide
      scrollContainer.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  // Scroll automatique vers le bas quand de nouveaux messages sont ajoutés
  useEffect(() => {
    // Fonction pour effectuer le scroll automatique
    const scrollToBottom = () => {
      // Essayer d'abord avec le conteneur de messages s'il existe
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
      }
      // Sinon utiliser le conteneur principal du side panel
      else if (sidePanelRef.current) {
        sidePanelRef.current.scrollTop = sidePanelRef.current.scrollHeight;
      }
    };

    // Vérifier si le panel est ouvert et qu'il y a des messages
    if (isExpanded && messages.length > 0) {
      // Délai court pour s'assurer que le DOM est mis à jour
      setTimeout(() => {
        scrollToBottom();
      }, 100);
      
      // Délai supplémentaire pour s'assurer que le contenu est entièrement rendu
      setTimeout(() => {
        scrollToBottom();
      }, 300);
      
      // Scroll fluide final
      setTimeout(() => {
        scrollToBottomSmooth();
      }, 500);
    }
  }, [messages, isExpanded]);

  // Scroll automatique spécifique quand le panel s'ouvre
  useEffect(() => {
    if (isExpanded && messages.length > 0) {
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        } else if (sidePanelRef.current) {
          sidePanelRef.current.scrollTop = sidePanelRef.current.scrollHeight;
        }
        // Scroll fluide après ouverture
        setTimeout(() => {
          scrollToBottomSmooth();
        }, 100);
      }, 200);
    }
  }, [isExpanded]);

  // Scroll automatique quand des commentaires sont ajoutés
  useEffect(() => {
    if (comments && comments.length > 0 && isExpanded) {
      // Délai plus long pour s'assurer que tous les commentaires sont traités
      setTimeout(() => {
        scrollToBottomSmooth();
      }, 500);
    }
  }, [comments, isExpanded]);

  // Nouvelle fonction pour générer l'analyse stratégique automatiquement
  const generateStrategicAnalysis = async () => {
    try {
      console.log("Starting automatic strategic analysis...");
      
      // Activer le spinner de chargement
      setIsStrategicAnalysisLoading(true);

      // Préparer les données complètes pour l'analyse stratégique
      const analysisData = {
        surface: propertyData?.surface || propertyData?.habitableSurface || 120,
        epcScore: propertyData?.epcScore || 'A_plus',
        heatingType: propertyData?.heatingType || 'ELECTRIC',
        bedrooms: propertyData?.bedrooms || propertyData?.bedroomCount || 3,
        bathrooms: propertyData?.bathrooms || propertyData?.bathroomCount || 1,
        buildingConstructionYear: propertyData?.buildingConstructionYear || propertyData?.constructionYear || 2000,
        municipality: propertyData?.municipality || propertyData?.locality || 'Antwerpen',
        province: propertyData?.province || 'Antwerpen',
        hasGarden: propertyData?.hasGarden || false,
        hasBalcony: propertyData?.hasBalcony || false,
        hasParking: propertyData?.hasParking || false,
        hasElevator: propertyData?.hasElevator || false,
        buildingCondition: propertyData?.buildingCondition || 'AS NEW',
        kitchenType: propertyData?.kitchenType || 'HYPER EQUIPPED',
        // Données ESG
        esgScores: esgData?.esg_scores || {},
        esgCompliance: esgData?.compliance_status || {},
        financialImpact: esgData?.financial_impact || {}
      };

      console.log("Sending strategic analysis request with data:", analysisData);

      // Utiliser l'API Chat pour générer une analyse stratégique enrichie
      const prompt = `Generate a comprehensive strategic analysis for this Belgian real estate investment. Follow this EXACT structure with these specific headers only:

# Strategic Analysis – ${analysisData.municipality} Property Investment

## Investment Positioning
Based on ESG scores (Environmental: ${analysisData.esgScores.environmental || 'N/A'}/10, Social: ${analysisData.esgScores.social || 'N/A'}/10, Governance: ${analysisData.esgScores.governance || 'N/A'}/10), analyze the investment potential.

## Market Context
Analyze the ${analysisData.municipality} market, construction year ${analysisData.buildingConstructionYear}, and EPC rating ${analysisData.esgScore} positioning. Include specific insights about the Antwerp market dynamics, rental demand, and property value trends.

## Short-term Actions (0-6 months)
Provide immediate improvement opportunities, quick wins for value enhancement, and priority maintenance items.

## Medium-term Strategy (6-24 months)  
Cover major improvement projects, market positioning optimization, and energy efficiency upgrades.

## Long-term Vision (2+ years)
Outline future-proofing strategies, regulatory compliance preparation, and portfolio expansion considerations.

## Risk Assessment
Evaluate potential investment risks and provide specific mitigation strategies. Focus on:
- Market volatility and economic factors
- Property-specific risks (age, condition, energy efficiency)
- ESG compliance risks and regulatory changes
- Financial risks (interest rates, financing, liquidity)
- Operational risks (maintenance, vacancy, tenant issues)
- Climate and environmental risks

IMPORTANT: Use ONLY these exact section headers. Do not create additional sections or repeat any section. 

Property details: Surface ${analysisData.surface}m², ${analysisData.bedrooms} bedrooms, ${analysisData.buildingCondition} condition, ${analysisData.heatingType} heating.`;

      const response = await axios.post(CHAT_API_URL, {
        messages: [
          {
            role: "system",
            content: "You are a senior Belgian real estate investment strategist with 20+ years experience. Provide detailed, actionable strategic analysis focusing on ESG compliance, market positioning, and investment optimization. Use professional language with specific Belgian market insights. For the Risk Assessment section, provide concrete, specific risks with practical mitigation strategies. Avoid generic statements and focus on actionable advice."
          },
          {
            role: "user", 
            content: prompt
          }
        ]
      });

      // Traiter la réponse et la diviser en sections
      const analysisText = response.data.response || "Strategic analysis completed.";
      
      // Nettoyer complètement tout message de chargement et de statut
      let cleanedText = analysisText
        .replace(/^.*?Generating.*?$/gmi, '')
        .replace(/^.*?Analyzing.*?$/gmi, '')
        .replace(/^.*?Strategic.*?in progress.*?$/gmi, '')
        .replace(/^.*?Market.*?in progress.*?$/gmi, '')
        .replace(/^.*?ESG.*?assessment.*?$/gmi, '')
        .replace(/^.*?Investment.*?recommendations.*?$/gmi, '')
        .replace(/^.*?Strategic.*?action.*?items.*?$/gmi, '')
        .replace(/^.*?complete.*?$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*.*?(progress|assessment|recommendations|items|complete|action).*?$/gmi, '')
        .replace(/^\s*Investment recommendations\s*$/gmi, '')
        .replace(/^\s*Strategic action items\s*$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*Investment recommendations\s*$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*Strategic action items\s*$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*Property shows strong investment potential\s*$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*ESG compliance aligned with market trends\s*$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*Long-term value optimization identified\s*$/gmi, '')
        .replace(/^\s*[\•\-\*]\s*Recommended next steps available\s*$/gmi, '')
        .replace(/^\s*Property shows strong investment potential\s*$/gmi, '')
        .replace(/^\s*ESG compliance aligned with market trends\s*$/gmi, '')
        .replace(/^\s*Long-term value optimization identified\s*$/gmi, '')
        .replace(/^\s*Recommended next steps available\s*$/gmi, '')
        .trim();
      
      // Supprimer les lignes vides multiples
      cleanedText = cleanedText.replace(/\n\s*\n\s*\n/g, '\n\n');
      
      // Diviser l'analyse en sections basées sur les headers markdown
      const sections = cleanedText.split(/(?=##?\s)/)
        .map(section => section.trim())
        .filter(section => {
          // Filtrage minimal : supprimer seulement les sections vraiment vides
          if (!section || section.length <= 5) return false;
          if (section === '#') return false;
          if (section.match(/^\s*#+\s*$/)) return false;
          
          // Filtrer SEULEMENT les messages de chargement explicites
          const lowerSection = section.toLowerCase();
          if (lowerSection.includes('generating') && lowerSection.includes('progress')) return false;
          if (lowerSection.includes('analyzing') && lowerSection.includes('progress')) return false;
          if (lowerSection === 'investment recommendations') return false;
          if (lowerSection === 'strategic action items') return false;
          if (lowerSection === 'property shows strong investment potential') return false;
          if (lowerSection === 'esg compliance aligned with market trends') return false;
          if (lowerSection === 'long-term value optimization identified') return false;
          if (lowerSection === 'recommended next steps available') return false;
          
          return true;
        })
        .filter((section, index, array) => {
          // Éviter les doublons en comparant les titres des sections avec normalisation
          const sectionTitle = section.match(/##?\s*([^\n]+)/)?.[1]?.toLowerCase().trim();
          if (!sectionTitle) return true;
          
          // Normaliser les titres pour détecter les variantes
          const normalizedTitle = sectionTitle
            .replace(/\s*\([^)]*\)\s*/g, '') // Supprimer les parenthèses (6-24 months), (2+ years)
            .replace(/[^\w\s]/g, '') // Supprimer la ponctuation
            .replace(/\s+/g, ' ') // Normaliser les espaces
            .trim();
          
          // Filtrer spécifiquement les doublons du titre principal "Strategic Analysis"
          if (normalizedTitle.includes('strategic analysis') && normalizedTitle.includes('property investment')) {
            // Vérifier si on a déjà une section avec ce titre
            const hasExistingStrategicAnalysis = array.slice(0, index).some(prevSection => {
              const prevTitle = prevSection.match(/##?\s*([^\n]+)/)?.[1]?.toLowerCase().trim();
              if (!prevTitle) return false;
              
              const prevNormalizedTitle = prevTitle
                .replace(/\s*\([^)]*\)\s*/g, '')
                .replace(/[^\w\s]/g, '')
                .replace(/\s+/g, ' ')
                .trim();
              
              return prevNormalizedTitle.includes('strategic analysis') && prevNormalizedTitle.includes('property investment');
            });
            
            if (hasExistingStrategicAnalysis) {
              console.log(`Filtering duplicate Strategic Analysis title: "${sectionTitle}"`);
              return false;
            }
          }
          
          // Vérifier si une section similaire existe déjà (logique générale)
          const isDuplicate = array.slice(0, index).some(prevSection => {
            const prevTitle = prevSection.match(/##?\s*([^\n]+)/)?.[1]?.toLowerCase().trim();
            if (!prevTitle) return false;
            
            const prevNormalizedTitle = prevTitle
              .replace(/\s*\([^)]*\)\s*/g, '')
              .replace(/[^\w\s]/g, '')
              .replace(/\s+/g, ' ')
              .trim();
            
            return prevNormalizedTitle === normalizedTitle;
          });
          
          return !isDuplicate;
        })
        .sort((a, b) => {
          // Trier les sections selon un ordre logique prédéfini
          const order = [
            'strategic analysis',
            'investment positioning',
            'market context',
            'short-term actions',
            'medium-term strategy',
            'long-term vision',
            'risk assessment'
          ];
          
          const getTitleOrder = (section) => {
            const title = section.match(/##?\s*([^\n]+)/)?.[1]?.toLowerCase().trim();
            if (!title) return 999;
            
            const normalizedTitle = title
              .replace(/\s*\([^)]*\)\s*/g, '')
              .replace(/[^\w\s]/g, '')
              .replace(/\s+/g, ' ')
              .trim();
            
            // Titre principal "Strategic Analysis" en premier
            if (normalizedTitle.includes('strategic analysis') && normalizedTitle.includes('property investment')) {
              return -1; // Toujours en premier
            }
            
            const orderIndex = order.findIndex(orderTitle => 
              normalizedTitle.includes(orderTitle) || orderTitle.includes(normalizedTitle)
            );
            
            return orderIndex === -1 ? 999 : orderIndex;
          };
          
          return getTitleOrder(a) - getTitleOrder(b);
        });
      
      // Désactiver le spinner avant d'ajouter les messages
      setIsStrategicAnalysisLoading(false);
      
      // Marquer l'analyse comme générée SEULEMENT si on a des sections valides
      if (sections.length > 0) {
        setStrategicAnalysisGenerated(true);
      }
      
      // Ajouter les sections comme messages séparés
      sections.forEach((section, index) => {
        // Vérifications basiques avant d'ajouter un message
        const cleanSection = section.trim();
        if (!cleanSection || cleanSection.length <= 5) return;
        
        // Filtrer SEULEMENT les messages de chargement explicites
        const lowerSection = cleanSection.toLowerCase();
        if (lowerSection.includes('generating') && lowerSection.includes('progress')) return;
        if (lowerSection.includes('analyzing') && lowerSection.includes('progress')) return;
        if (lowerSection === 'investment recommendations') return;
        if (lowerSection === 'strategic action items') return;
        if (lowerSection === 'property shows strong investment potential') return;
        if (lowerSection === 'esg compliance aligned with market trends') return;
        if (lowerSection === 'long-term value optimization identified') return;
        if (lowerSection === 'recommended next steps available') return;
        
        // Vérifier que le contenu formaté ne sera pas vide
        const testFormatted = formatMessageText(cleanSection);
        if (!testFormatted || !testFormatted.__html || testFormatted.__html.trim() === '') return;
        
        setTimeout(() => {
          setMessages(prev => [...prev, {
            from: "agent", 
            text: cleanSection,
            type: "prediction",
            subtype: "strategic-title",
            timestamp: new Date().toISOString()
          }]);
          
          // Scroll automatique après chaque section ajoutée
          setTimeout(() => {
            scrollToBottomSmooth();
          }, 100);
        }, index * 1000); // Délai progressif pour affichage fluide
      });

    } catch (err) {
      console.error("Strategic analysis error:", err.response?.data || err.message || err);
      
      // Désactiver le spinner en cas d'erreur
      setIsStrategicAnalysisLoading(false);
      
      setMessages(prev => [
        ...prev,
        { 
          from: "agent", 
          text: `❌ **Strategic Analysis Error**\n\nUnable to generate strategic analysis: ${err.response?.data?.detail || err.message || 'API unavailable'}. The ESG analysis is still available in the right panel.`, 
          type: "prediction",
          subtype: "strategic-title",
          timestamp: new Date().toISOString()
        }
      ]);
    }
  };

  const clearChatHistory = () => {
    setMessages([
      { from: "agent", text: "Hello! How can I assist you today?" }
    ]);
    // Reset strategic analysis state
    setStrategicAnalysisGenerated(false);
    setIsStrategicAnalysisLoading(false);
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
    
    // Scroll automatique après l'envoi du message utilisateur
    setTimeout(() => {
      scrollToBottomSmooth();
    }, 100);

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
      
      // Déclencher le scroll automatique après la réponse
      setTimeout(() => {
        scrollToBottomSmooth();
      }, 100);
    } catch (err) {
      console.error("Chat error:", err.response?.data || err.message || err);
      setMessages(prev => [
        ...prev,
        { from: "agent", text: "Sorry, I couldn't reach the assistant.", timestamp: new Date().toISOString() }
      ]);
      
      // Déclencher le scroll automatique même en cas d'erreur
      setTimeout(() => {
        scrollToBottomSmooth();
      }, 100);
    }
  };

  // Fonction pour formater les messages utilisateur (sans couleurs, texte blanc)
  const formatUserMessage = (text) => {
    if (!text) return { __html: '' };
    
    let formattedText = text.toString().trim();
    
    // Vérifier si le message est vide
    if (!formattedText) {
      return { __html: '' };
    }
    
    // Supprimer les emojis pour un style plus propre
    formattedText = formattedText.replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|[\u{1F900}-\u{1F9FF}]|[\u{1F018}-\u{1F0FF}]/gu, '');
    
    // Convertir les titres markdown en titres simples sans couleurs (tout en blanc)
    formattedText = formattedText.replace(/(^|[\s\n])#{4}\s+([^\n]+)/gm, '$1<h5 style="margin: 10px 0 5px 0; font-weight: 600; color: white; font-size: 13px;">$2</h5>');
    formattedText = formattedText.replace(/(^|[\s\n])#{3}\s+([^\n]+)/gm, '$1<h4 style="margin: 12px 0 6px 0; font-weight: bold; color: white; font-size: 14px;">$2</h4>');
    formattedText = formattedText.replace(/(^|[\s\n])#{2}\s+([^\n]+)/gm, '$1<h3 style="margin: 14px 0 8px 0; font-weight: bold; color: white; font-size: 15px;">$2</h3>');
    formattedText = formattedText.replace(/(^|[\s\n])#{1}\s+([^\n]+)/gm, '$1<h2 style="margin: 16px 0 10px 0; font-weight: bold; color: white; font-size: 17px;">$2</h2>');
    
    // Convertir **texte** en texte gras blanc
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong style="color: white;">$1</strong>');
    
    // Traiter les bullet points avec alignement correct (en blanc)
    formattedText = formattedText.replace(/^[•\+\-]\s*(.*?)$/gm, '<div style="margin: 4px 0; padding-left: 16px; color: white; line-height: 1.4; position: relative;"><span style="position: absolute; left: 0; color: white; font-weight: bold;">•</span>$1</div>');
    
    // Convertir les sauts de ligne en <br/>
    formattedText = formattedText.replace(/\n(?!\s*<)/g, '<br/>');
    
    // Nettoyer les <br/> en trop
    formattedText = formattedText.replace(/(<br\/>){3,}/g, '<br/><br/>');
    formattedText = formattedText.replace(/<br\/>\s*(<h[1-6])/g, '$1');
    formattedText = formattedText.replace(/(<\/h[1-6]>)\s*<br\/>/g, '$1');
    formattedText = formattedText.replace(/<br\/>\s*(<div)/g, '$1');
    formattedText = formattedText.replace(/(<\/div>)\s*<br\/>/g, '$1');
    
    // Nettoyer les espaces en début et fin
    formattedText = formattedText.trim();
    
    return { __html: formattedText };
  };

  // Fonction pour convertir le markdown simple (**texte**) en HTML et supprimer les emojis
  const formatMessageText = (text) => {
    if (!text) return { __html: '' };
    
    let formattedText = text.toString().trim();
    
    // Vérifier si le message est vide ou ne contient que des espaces/caractères spéciaux
    if (!formattedText || formattedText.match(/^\s*[#\s]*$/)) {
      return { __html: '' };
    }
    
    // FILTRAGE CIBLÉ RÉDUIT : supprimer SEULEMENT les messages de chargement explicites
    const lowerText = formattedText.toLowerCase();
    if (lowerText.includes('generating') && lowerText.includes('progress')) return { __html: '' };
    if (lowerText.includes('analyzing') && lowerText.includes('progress')) return { __html: '' };
    if (lowerText === 'investment recommendations') return { __html: '' };
    if (lowerText === 'strategic action items') return { __html: '' };
    
    // FILTRAGE SPÉCIFIQUE POUR LES MESSAGES DE STATUT SEULEMENT
    if (lowerText === 'property shows strong investment potential') return { __html: '' };
    if (lowerText === 'esg compliance aligned with market trends') return { __html: '' };
    if (lowerText === 'long-term value optimization identified') return { __html: '' };
    if (lowerText === 'recommended next steps available') return { __html: '' };
    
    // 0. SUPPRIMER TOUS LES EMOJIS ET ICÔNES (en premier)
    formattedText = formattedText.replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|[\u{1F900}-\u{1F9FF}]|[\u{1F018}-\u{1F0FF}]/gu, '');
    // Supprimer les caractères spéciaux couramment utilisés comme icônes sauf ceux utiles
    formattedText = formattedText.replace(/[🔄📊🏠💰⚡🌱📈🧠⚠️✅🤖💡👋🏷️📅🔥🏗️ℹ️📋❌🎯]/g, '');
    // Nettoyer les espaces multiples résultant de la suppression d'emojis
    formattedText = formattedText.replace(/\s{2,}/g, ' ');
    
    // 1. SUPPRIMER LES # QUI TRAÎNENT EN FIN DE LIGNE ET LIGNES
    formattedText = formattedText.replace(/^\s*#\s*$/gm, ''); // Supprimer les lignes avec juste #
    formattedText = formattedText.replace(/\s+#\s*$/gm, ''); // Supprimer les # en fin de ligne
    
    // 2. Supprimer les séparateurs markdown (barres horizontales) EN PREMIER
    formattedText = formattedText.replace(/---+/g, '');
    formattedText = formattedText.replace(/^\s*-{3,}\s*$/gm, '');
    formattedText = formattedText.replace(/^\s*\*{3,}\s*$/gm, '');
    formattedText = formattedText.replace(/^\s*_{3,}\s*$/gm, '');
    
    // 3. Gérer les titres avec numéros qui peuvent être collés au texte précédent
    formattedText = formattedText.replace(/(\w+)(\s*#{3}\s*\d+\.\s+[^\n]+)/g, '$1<br/>$2');
    formattedText = formattedText.replace(/(\w+)(\s*#{2}\s*\d+\.\s+[^\n]+)/g, '$1<br/>$2');
    formattedText = formattedText.replace(/(\w+)(\s*#{1}\s*\d+\.\s+[^\n]+)/g, '$1<br/>$2');
    
    // 4. Convertir les titres markdown # ## ### #### en titres HTML propres
    // Ordre important : du plus spécifique au moins spécifique
    // Gérer les cas où les titres peuvent être au milieu du texte ou après des espaces
    formattedText = formattedText.replace(/(^|[\s\n])#{4}\s+([^\n]+)/gm, '$1<h5 style="margin: 10px 0 5px 0; font-weight: 600; color: #1565c0; font-size: 13px;">$2</h5>');
    formattedText = formattedText.replace(/(^|[\s\n])#{3}\s+([^\n]+)/gm, '$1<h4 style="margin: 12px 0 6px 0; font-weight: bold; color: #2563eb; font-size: 14px;">$2</h4>');
    formattedText = formattedText.replace(/(^|[\s\n])#{2}\s+([^\n]+)/gm, '$1<h3 style="margin: 14px 0 8px 0; font-weight: bold; color: #8b5a2b; font-size: 15px;">$2</h3>');
    formattedText = formattedText.replace(/(^|[\s\n])#{1}\s+([^\n]+)/gm, '$1<h2 style="margin: 16px 0 10px 0; font-weight: bold; color: #6a1b9a; font-size: 17px; border-bottom: 1px solid #e0e0e0; padding-bottom: 3px;">$2</h2>');
    
    // 4.1 Gérer les titres avec numéros qui peuvent être au milieu du texte
    formattedText = formattedText.replace(/(?:^|\n)#{3}\s*(\d+)\.\s+([^\n]+)/gm, '<h4 style="margin: 12px 0 6px 0; font-weight: bold; color: #2563eb; font-size: 14px;">$1. $2</h4>');
    formattedText = formattedText.replace(/(?:^|\n)#{2}\s*(\d+)\.\s+([^\n]+)/gm, '<h3 style="margin: 14px 0 8px 0; font-weight: bold; color: #8b5a2b; font-size: 15px;">$1. $2</h3>');
    formattedText = formattedText.replace(/(?:^|\n)#{1}\s*(\d+)\.\s+([^\n]+)/gm, '<h2 style="margin: 16px 0 10px 0; font-weight: bold; color: #6a1b9a; font-size: 17px; border-bottom: 1px solid #e0e0e0; padding-bottom: 3px;">$1. $2</h2>');
    
    // 5. Convertir **texte** en <strong>texte</strong> APRÈS les titres pour éviter les conflits
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 6. Traiter les puces différemment selon leur contexte
    // D'abord traiter les puces qui suivent "How:" ou "Why:"
    formattedText = formattedText.replace(/(Why:|How:)\s*\n?\s*-\s*(.+)/gi, '$1<br/><div style="margin: 4px 0 4px 16px; padding: 0; line-height: 1.4;"><span style="color: #6a1b9a; font-weight: bold;">•</span> $2</div>');
    
    // Ensuite traiter les puces normales en début de ligne
    formattedText = formattedText.replace(/^[•\+\-]\s*(.*?)$/gm, '<div style="margin: 2px 0 2px 16px; padding: 0; line-height: 1.4; position: relative;"><span style="position: absolute; left: -12px; color: #6a1b9a; font-weight: bold;">•</span>$1</div>');
    
    // 7. Nettoyer les sauts de ligne excessifs
    formattedText = formattedText.replace(/\n\s*\n\s*\n/g, '\n\n');
    
    // 8. Convertir les sauts de ligne simples en <br/> mais éviter autour des balises HTML
    formattedText = formattedText.replace(/\n(?!\s*<)/g, '<br/>');
    
    // 9. Nettoyer les <br/> en trop et les lignes vides
    formattedText = formattedText.replace(/(<br\/>){3,}/g, '<br/><br/>');
    formattedText = formattedText.replace(/<br\/>\s*(<h[1-6])/g, '$1');
    formattedText = formattedText.replace(/(<\/h[1-6]>)\s*<br\/>/g, '$1');
    formattedText = formattedText.replace(/<br\/>\s*(<div)/g, '$1');
    formattedText = formattedText.replace(/(<\/div>)\s*<br\/>/g, '$1');
    
    // 10. Supprimer les lignes vides restantes
    formattedText = formattedText.replace(/<br\/>\s*<br\/>\s*<br\/>/g, '<br/><br/>');
    
    // 11. Nettoyer les espaces en début et fin
    formattedText = formattedText.trim();
    
    // Dernière vérification : si le contenu final est vide, retourner vide
    const finalContent = formattedText.replace(/<[^>]*>/g, '').trim();
    if (!finalContent) {
      return { __html: '' };
    }
    
    return { __html: formattedText };
  };

  // Gestion du redimensionnement améliorée
  const handleMouseDown = (e) => {
    setIsResizing(true);
    setStartX(e.clientX);
    setStartWidth(panelWidth);
    e.preventDefault();
  };

  const handleMouseMove = (e) => {
    if (!isResizing) return;
    
    const deltaX = e.clientX - startX;
    const newWidth = startWidth + deltaX;
    const minWidth = 10; // Largeur minimale très petite
    
    // Permettre le redimensionnement de 10px jusqu'à toute la largeur
    if (newWidth >= minWidth) {
      setPanelWidth(newWidth);
    }
  };

  const handleMouseUp = () => {
    setIsResizing(false);
  };

  // Effet pour gérer les événements de souris globaux pendant le redimensionnement
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    // Cleanup function pour s'assurer que les event listeners sont supprimés
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing, startX, startWidth]);

  return (
    <>
      {/* Onglet visible pour rouvrir le panel */}
      <div
        className={`sidepanel-tab ${isExpanded ? 'hidden' : ''}`}
        onClick={onToggle}
        title="Open AI Chat Assistant"
      >
        <span className="sidepanel-tab-text">CHAT</span>
      </div>

      <aside 
        ref={panelRef}
        className={`sidepanel ${isExpanded ? "open" : ""}`}
        style={{ 
          width: `${panelWidth}px`,
          left: isExpanded ? 0 : `-${panelWidth + 40}px`
        }}
      >
        {isExpanded && (
          <>
            {/* Handle de redimensionnement */}
            <div 
              className="resize-handle"
              onMouseDown={handleMouseDown}
            />
            
            <div className="sidepanel-header">
              {panelWidth >= 200 && <h3>Profile: {user.profile}</h3>}
              <button className="close-btn" onClick={onClose} aria-label="Close Side Panel">
                &times;
              </button>
            </div>

            {/* Spinner pour l'analyse stratégique en cours - Position fixe entre Profile et AI Chat */}
            {isStrategicAnalysisLoading && (
              <div className="strategic-analysis-loading-fixed">
                <div className="spinner"></div>
                <span className="loading-text">
                  ESG Strategic Analysis in progress...
                </span>
              </div>
            )}

            <div ref={sidePanelRef} className="sidepanel-content">
              <section className="chat-section">
                <div className="chat-header">
                  <h4>AI Chat Assistant</h4>
                  <div className="chat-header-buttons">

                    <button
                      onClick={() => {
                        // Éviter les clics multiples quand l'analyse est déjà en cours
                        if (isStrategicAnalysisLoading) {
                          console.log("Strategic analysis already in progress, ignoring click");
                          return;
                        }
                        
                        console.log("Manual strategic analysis trigger");
                        generateStrategicAnalysis();
                        // Ne pas définir setStrategicAnalysisGenerated(true) ici
                        // car cela sera fait dans generateStrategicAnalysis() seulement en cas de succès
                      }}
                      className="strategic-analysis-btn"
                      title="Generate strategic analysis"
                      disabled={isStrategicAnalysisLoading}
                    >
                      {isStrategicAnalysisLoading ? 'Generating...' : 'ESG Strategic Analysis'}
                    </button>
                    <button
                      onClick={clearChatHistory}
                      className="clear-chat-btn"
                      title="Clear chat history"
                    >
                      Clear
                    </button>
                  </div>
                </div>

                <div className="chat-messages" ref={messagesContainerRef}>
                  {messages
                    .filter(msg => {
                      // Filtrer les messages vides ou qui ne contiennent que des espaces/caractères spéciaux
                      const cleanText = msg.text ? msg.text.replace(/[#\s\n\r]/g, '') : '';
                      if (cleanText.length === 0) return false;
                      
                      // Filtrer SEULEMENT les messages de chargement explicites
                      const lowerText = msg.text.toLowerCase();
                      if (lowerText.includes('generating') && lowerText.includes('progress')) return false;
                      if (lowerText.includes('analyzing') && lowerText.includes('progress')) return false;
                      if (lowerText === 'investment recommendations') return false;
                      if (lowerText === 'strategic action items') return false;
                      
                      // FILTRES SPÉCIFIQUES POUR LES MESSAGES DE STATUT SEULEMENT
                      if (lowerText === 'property shows strong investment potential') return false;
                      if (lowerText === 'esg compliance aligned with market trends') return false;
                      if (lowerText === 'long-term value optimization identified') return false;
                      if (lowerText === 'recommended next steps available') return false;
                      
                      return true;
                    })
                    .map((msg, idx) => {
                      // Pour les messages utilisateur, utiliser un formatage simple sans couleurs
                      if (msg.from === "user") {
                        const userContent = formatUserMessage(msg.text);
                        if (!userContent || !userContent.__html || userContent.__html.trim() === '') {
                          return null;
                        }
                        return (
                          <div
                            key={idx}
                            className="message user-msg"
                            dangerouslySetInnerHTML={userContent}
                          />
                        );
                      }
                      
                      // Pour les messages agent, utiliser le formatage avec couleurs
                      const formattedContent = formatMessageText(msg.text);
                      if (!formattedContent || !formattedContent.__html || formattedContent.__html.trim() === '') {
                        return null; // Ne pas rendre les messages vides
                      }
                      
                      return (
                        <div
                          key={idx}
                          className={`message agent-msg ${msg.type === "prediction" ? "prediction-msg" : ""} ${msg.subtype === "prediction-title" ? "prediction-title" 
                            : ""} ${msg.subtype === "esg-title" ? "esg-title" : ""} ${msg.subtype === "strategic-title" ? "strategic-title" : ""} ${msg.subtype === "model-info" ? "model-info" : ""} ${msg.subtype === "prediction-comment" ? "prediction-comment" : ""}`}
                          dangerouslySetInnerHTML={formattedContent}
                        />
                      );
                    })
                    .filter(Boolean)} {/* Supprimer les éléments null */}

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
