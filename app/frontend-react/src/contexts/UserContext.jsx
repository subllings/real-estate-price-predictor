import React, { createContext, useContext, useState, useEffect } from 'react';

const UserContext = createContext();

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Cosmos DB API base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8010';

  // Load user from localStorage on app start
  useEffect(() => {
    const savedUser = localStorage.getItem('esg_user');
    if (savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        setUser(userData);
        loadUserProfile(userData.user_id);
      } catch (err) {
        console.error('Error parsing saved user data:', err);
        localStorage.removeItem('esg_user');
      }
    }
    setLoading(false);
  }, []);

  // Load user profile from Cosmos DB
  const loadUserProfile = async (userId) => {
    try {
      setError(null);
      const response = await fetch(`${API_BASE_URL}/api/users/${userId}`);
      
      if (response.ok) {
        const profile = await response.json();
        setUserProfile(profile);
      } else if (response.status === 404) {
        // User doesn't exist, create default profile
        await createDefaultUserProfile(userId);
      } else {
        throw new Error('Failed to load user profile');
      }
    } catch (err) {
      console.error('Error loading user profile:', err);
      setError('Failed to load user profile');
      
      // Fallback to default profile
      setUserProfile(getDefaultProfile(userId));
    }
  };

  // Create default user profile in Cosmos DB
  const createDefaultUserProfile = async (userId) => {
    try {
      const defaultProfile = getDefaultProfile(userId);
      
      const response = await fetch(`${API_BASE_URL}/api/users`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(defaultProfile)
      });

      if (response.ok) {
        const profile = await response.json();
        setUserProfile(profile);
      } else {
        throw new Error('Failed to create user profile');
      }
    } catch (err) {
      console.error('Error creating user profile:', err);
      // Fallback to local default profile
      setUserProfile(getDefaultProfile(userId));
    }
  };

  // Get default profile based on email domain or user preferences
  const getDefaultProfile = (userId) => {
    const email = user?.email || '';
    let userRole = 'Property Owner'; // default
    
    // Determine role based on email or other factors
    if (email.includes('investor') || email.includes('investment')) {
      userRole = 'Investor';
    } else if (email.includes('agent') || email.includes('broker')) {
      userRole = 'Real Estate Agent';
    } else if (email.includes('developer')) {
      userRole = 'Developer';
    }

    return {
      id: userId,
      user_id: userId,
      email: email,
      auth_type: user?.auth_type || 'local',
      user_role: userRole,
      suggested_questions: getQuestionsForRole(userRole),
      created_at: new Date().toISOString(),
      last_login: new Date().toISOString()
    };
  };

  // Get role-specific questions
  const getQuestionsForRole = (role) => {
    const questionSets = {
      'Investor': [
        'What ESG risks exist for EPC Class F properties?',
        'Which Belgian subsidies maximize ROI for renovations?',
        'What renovations boost resale value most?',
        'How do 2030 regulations affect rental property investments?',
        'What are the compliance costs for energy upgrades?'
      ],
      'Property Owner': [
        'How much will EPC Class E to B renovation cost?',
        'What grants are available in my region?',
        'When should I renovate to avoid 2030 penalties?',
        'What energy improvements add most value?',
        'How do I calculate renovation ROI?'
      ],
      'Real Estate Agent': [
        'How do I explain EPC impact to clients?',
        'What renovation advice should I give sellers?',
        'How do 2030 deadlines affect property marketing?',
        'What ESG factors affect property valuations?',
        'How do I identify ESG investment opportunities?'
      ],
      'Developer': [
        'What are the new construction ESG requirements?',
        'How do I optimize ESG scores for new projects?',
        'What sustainable materials qualify for incentives?',
        'How do ESG factors affect project financing?',
        'What are future-proof building standards?'
      ]
    };

    return questionSets[role] || questionSets['Property Owner'];
  };

  // Login function
  const login = async (userData) => {
    try {
      setLoading(true);
      setError(null);

      // Save user data
      setUser(userData);
      localStorage.setItem('esg_user', JSON.stringify(userData));

      // Load or create user profile
      await loadUserProfile(userData.user_id);

      return true;
    } catch (err) {
      setError('Login failed');
      console.error('Login error:', err);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    setUser(null);
    setUserProfile(null);
    setError(null);
    localStorage.removeItem('esg_user');
  };

  // Update user profile
  const updateUserProfile = async (updates) => {
    try {
      if (!userProfile) return false;

      const updatedProfile = { ...userProfile, ...updates };
      
      const response = await fetch(`${API_BASE_URL}/api/users/${userProfile.user_id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedProfile)
      });

      if (response.ok) {
        const profile = await response.json();
        setUserProfile(profile);
        return true;
      } else {
        throw new Error('Failed to update profile');
      }
    } catch (err) {
      console.error('Error updating user profile:', err);
      setError('Failed to update profile');
      return false;
    }
  };

  const value = {
    user,
    userProfile,
    loading,
    error,
    login,
    logout,
    updateUserProfile,
    isAuthenticated: !!user
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
};

export default UserContext;
