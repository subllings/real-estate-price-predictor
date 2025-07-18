// Test pour vérifier si le bouton est cliquable
console.log('🔍 Testing button click handler...');

// Fonction pour vérifier l'état du bouton
const checkButtonState = () => {
  // Chercher le bouton par son texte
  const button = [...document.querySelectorAll('button')].find(btn => 
    btn.textContent.includes('Analyze Price & ESG')
  );
  
  if (button) {
    console.log('✅ Button found:', button);
    console.log('🔍 Button disabled?', button.disabled);
    console.log('🔍 Button classList:', button.classList.toString());
    console.log('🔍 Button textContent:', button.textContent);
    console.log('🔍 Button onclick:', button.onclick);
    console.log('🔍 Button addEventListener events:', button);
    
    // Essayer de cliquer manuellement
    console.log('🖱️ Attempting manual click...');
    button.click();
  } else {
    console.log('❌ Button not found!');
    console.log('🔍 All buttons on page:', [...document.querySelectorAll('button')].map(b => b.textContent));
  }
};

// Attendre que le DOM soit chargé
setTimeout(checkButtonState, 2000);

// Exporter pour utilisation dans la console
window.checkButtonState = checkButtonState;
