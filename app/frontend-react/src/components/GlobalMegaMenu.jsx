/**
 * Global Navigation Mega Menu with Admin Panel
 * Real Estate focused agent selection and navigation
 */
import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useUser } from '../contexts/UserContext';
import { LogOut, User } from 'lucide-react';

const GlobalMegaMenu = ({ onAdminToggle }) => {
  const [activeDropdown, setActiveDropdown] = useState(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();
  const { user, logout } = useUser();

  // Debug console pour forcer recompilation
  console.log('GlobalMegaMenu renderé avec props:', { onAdminToggle });

  const menuItems = [
    {
      id: 'home',
      label: 'Home',
      path: '/home',
      description: 'Real Estate AI Platform'
    },
    {
      id: 'predict',
      label: 'Price Predictor',
      path: '/',
      description: 'AI-powered real estate valuation'
    },
    {
      id: 'esg',
      label: 'ESG Agent',
      path: '/esg-agent',
      description: 'Sustainability & compliance advisor'
    },
    {
      id: 'documents',
      label: 'Documents RAG',
      path: '/documents',
      description: 'Intelligent document processing & analysis'
    },
    {
      id: 'training',
      label: 'Model Training',
      path: '/training',
      description: 'Azure ML training & optimization'
    },
    {
      id: 'admin',
      label: 'Admin Panel',
      action: 'admin',
      description: 'System monitoring & management'
    }
  ];

  // Real Estate focused agents only
  const relevantAgents = [
    { id: 'predict', name: 'Price Predictor', path: '/' },
    { id: 'esg', name: 'ESG Agent', path: '/esg-agent' },
    { id: 'training', name: 'Model Training', path: '/training' },
    { id: 'finance', name: 'Financial Insights', path: '/agent/finance' },
    { id: 'passive', name: 'Investment Analysis', path: '/agent/passive' }
  ];

  const handleItemClick = (item) => {
    console.log('handleItemClick called with:', item);
    if (item.action === 'admin') {
      console.log('Admin button clicked, calling onAdminToggle');
      onAdminToggle();
    }
  };

  return (
    <>
      {/* Main Navigation Bar */}
      <nav className="bg-gradient-to-r from-green-600 to-emerald-600 text-white shadow-lg relative z-30">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            {/* Logo/Brand */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-bold text-green-600">RE</span>
              </div>
              <span className="text-xl font-bold">RealEstate AI</span>
            </div>

            {/* Desktop Menu */}
            <div className="hidden md:flex items-center space-x-1">
              {menuItems.map(item => (
                <div key={item.id} className="relative group">
                  {item.path ? (
                    <Link
                      to={item.path}
                      className={`px-4 py-2 rounded-lg transition-all duration-200 hover:bg-white hover:bg-opacity-20 ${
                        location.pathname === item.path ? 'bg-white bg-opacity-30' : ''
                      }`}
                    >
                      {item.label}
                    </Link>
                  ) : (
                    <button
                      onClick={() => handleItemClick(item)}
                      className="px-4 py-2 rounded-lg transition-all duration-200 hover:bg-white hover:bg-opacity-20"
                    >
                      {item.label}
                    </button>
                  )}
                </div>
              ))}
              
              {/* User Menu */}
              {user && (
                <div className="flex items-center space-x-3 ml-6 pl-6 border-l border-white border-opacity-30">
                  <div className="flex items-center space-x-2">
                    <User size={16} />
                    <span className="text-sm font-medium">{user.name}</span>
                    <span className="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">
                      {user.role}
                    </span>
                  </div>
                  <button
                    onClick={logout}
                    className="px-3 py-1 rounded-lg transition-all duration-200 hover:bg-white hover:bg-opacity-20 flex items-center space-x-1"
                    title="Logout"
                  >
                    <LogOut size={14} />
                    <span className="text-sm">Logout</span>
                  </button>
                </div>
              )}
            </div>

            {/* Mobile menu button */}
            <button
              className="md:hidden p-2"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden bg-green-700 border-t border-green-500">
            <div className="px-4 py-3 space-y-2">
              <div className="text-lg font-medium text-green-100 mb-3">REAL ESTATE AI</div>

              {menuItems.map(item => (
                <div key={item.id}>
                  {item.path ? (
                    <Link
                      to={item.path}
                      className="block px-3 py-2 rounded-lg text-white hover:bg-green-600 transition-colors duration-200"
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      {item.label}
                    </Link>
                  ) : (
                    <button
                      onClick={() => {
                        handleItemClick(item);
                        setIsMobileMenuOpen(false);
                      }}
                      className="block w-full text-left px-3 py-2 rounded-lg text-white hover:bg-green-600 transition-colors duration-200"
                    >
                      {item.label}
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </nav>
    </>
  );
};

export default GlobalMegaMenu;

