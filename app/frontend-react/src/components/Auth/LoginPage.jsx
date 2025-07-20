import React, { useState } from 'react';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';

const LoginPage = ({ onLogin }) => {
  const [loginMode, setLoginMode] = useState('local'); // Default to 'local' since Google OAuth requires configuration
  const [localCredentials, setLocalCredentials] = useState({
    email: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Google OAuth configuration
  const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || 'your-google-client-id';

  const handleGoogleSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');
    
    try {
      // Decode the JWT token to get user info
      const userInfo = JSON.parse(atob(credentialResponse.credential.split('.')[1]));
      
      const userData = {
        user_id: userInfo.sub,
        email: userInfo.email,
        name: userInfo.name,
        auth_type: 'google',
        profile_picture: userInfo.picture
      };

      // Call parent component's login handler
      await onLogin(userData);
    } catch (err) {
      setError('Failed to authenticate with Google. Please try again.');
      console.error('Google auth error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleError = () => {
    setError('Google authentication failed. Please try again.');
  };

  const handleLocalLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Validate input
      if (!localCredentials.email || !localCredentials.password) {
        throw new Error('Please enter both email and password');
      }

      // Demo authentication - check against demo accounts
      const demoAccounts = [
        { email: 'demo@company.com', password: 'password123', name: 'Demo User', role: 'analyst' },
        { email: 'admin@company.com', password: 'admin123', name: 'Admin User', role: 'admin' },
        { email: 'investor@demo.com', password: 'demo123', name: 'Investment Manager', role: 'manager' },
        { email: 'owner@demo.com', password: 'demo123', name: 'Property Owner', role: 'owner' },
        { email: 'agent@demo.com', password: 'demo123', name: 'Real Estate Agent', role: 'consultant' }
      ];

      console.log('Attempting login with:', { 
        email: localCredentials.email, 
        passwordLength: localCredentials.password?.length 
      });

      const user = demoAccounts.find(
        account => account.email === localCredentials.email && account.password === localCredentials.password
      );

      if (!user) {
        console.log('No matching user found. Available accounts:', demoAccounts.map(acc => acc.email));
        throw new Error('Invalid credentials. Please check email and password or use demo accounts listed below.');
      }

      const userData = {
        user_id: `local_${user.email}`,
        email: user.email,
        name: user.name,
        role: user.role,
        auth_type: 'local'
      };

      // Call parent component's login handler
      await onLogin(userData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-400 to-blue-500 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-md p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Real Estate Agents Login
          </h1>
          <p className="text-gray-600">
            Access your personalized real estate advisor
          </p>
        </div>

        {/* Login Mode Toggle */}
        <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setLoginMode('google')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              loginMode === 'google'
                ? 'bg-white text-gray-800 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            Google OAuth
          </button>
          <button
            onClick={() => setLoginMode('local')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              loginMode === 'local'
                ? 'bg-white text-gray-800 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            Local Account
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Google Login */}
        {loginMode === 'google' && (
          <div className="space-y-4">
            {GOOGLE_CLIENT_ID !== 'demo-google-client-id' && GOOGLE_CLIENT_ID !== 'your-google-client-id' ? (
              <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
                <GoogleLogin
                  onSuccess={handleGoogleSuccess}
                  onError={handleGoogleError}
                  text="signin_with"
                  shape="rectangular"
                  theme="outline"
                  size="large"
                  width="100%"
                  disabled={loading}
                />
                <p className="text-xs text-gray-500 text-center">
                  Sign in with your Google account to access personalized real estate recommendations
                </p>
              </GoogleOAuthProvider>
            ) : (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <div className="text-center">
                  <p className="text-sm text-yellow-800 font-medium mb-2">
                    Google OAuth Not Configured
                  </p>
                  <p className="text-xs text-yellow-700 mb-3">
                    Google authentication requires a valid client ID. Please use Local Account login or contact your administrator.
                  </p>
                  <button
                    onClick={() => setLoginMode('local')}
                    className="bg-blue-500 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-600 transition-colors"
                  >
                    Switch to Local Account
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Local Login */}
        {loginMode === 'local' && (
          <form onSubmit={handleLocalLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email Address
              </label>
              <input
                type="email"
                value={localCredentials.email}
                onChange={(e) => setLocalCredentials({
                  ...localCredentials,
                  email: e.target.value
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your email"
                disabled={loading}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                type="password"
                value={localCredentials.password}
                onChange={(e) => setLocalCredentials({
                  ...localCredentials,
                  password: e.target.value
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your password"
                disabled={loading}
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {loading ? 'Signing In...' : 'Sign In'}
            </button>
            <p className="text-xs text-gray-500 text-center">
              Use your local account credentials to access the platform
            </p>
          </form>
        )}

        {/* Demo Account Info */}
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
          <h4 className="text-sm font-medium text-blue-800 mb-3">Available Demo Accounts</h4>
          <div className="text-xs text-blue-600 space-y-2">
            <div className="grid grid-cols-1 gap-1">
              <p><strong>Demo User:</strong> demo@company.com / password123</p>
              <p><strong>Admin User:</strong> admin@company.com / admin123</p>
              <p><strong>Investment Manager:</strong> investor@demo.com / demo123</p>
              <p><strong>Property Owner:</strong> owner@demo.com / demo123</p>
              <p><strong>Real Estate Agent:</strong> agent@demo.com / demo123</p>
            </div>
            <p className="text-blue-500 text-center mt-2 italic">
              Each role provides personalized question recommendations
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
