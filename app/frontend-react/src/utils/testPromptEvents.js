// Test script pour vérifier les événements de prompts
console.log('🧪 Testing prompt events...');

// Fonction pour tester l'envoi d'événements
const testPromptEvent = () => {
  console.log('📤 Sending test prompt event...');
  
  window.dispatchEvent(new CustomEvent('llmPromptSent', {
    detail: {
      type: 'TEST_PROMPT',
      prompt: 'This is a test prompt to verify the AdminPanel is listening correctly.',
      timestamp: new Date().toISOString(),
      metadata: {
        test: true,
        source: 'manual_test'
      }
    }
  }));
  
  console.log('✅ Test prompt event sent!');
};

// Fonction pour vérifier si l'AdminPanel écoute
const checkAdminPanelListeners = () => {
  console.log('🔍 Checking if AdminPanel is listening...');
  
  // Vérifier si il y a des listeners sur l'événement
  const hasListeners = window.addEventListener.toString().includes('llmPromptSent');
  console.log('AdminPanel listeners detected:', hasListeners);
  
  return hasListeners;
};

// Exporter les fonctions pour utilisation dans la console
window.testPromptEvent = testPromptEvent;
window.checkAdminPanelListeners = checkAdminPanelListeners;

// Test automatique
setTimeout(() => {
  console.log('🚀 Auto-testing prompt events...');
  testPromptEvent();
}, 1000);

export { testPromptEvent, checkAdminPanelListeners };
