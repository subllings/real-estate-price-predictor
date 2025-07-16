/**
 * Global Navigation Mega Menu with Admi  // Real Estate focused agents only
  const relevantAgents = [
    { id: 'predict', name: 'Price Predictor', path: '/' },
    { id: 'esg', name: 'ESG Agent', path: '/esg-agent' },
    { id: 'training', name: 'Model Training', path: '/training' },
    { id: 'finance', name: 'Financial Insights', path: '/agent/finance' },
    { id: 'passive', name: 'Investment Analysis', path: '/agent/passive' }
  ];
 * Visible on all pages for easy demo navigation
 */

import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const GlobalMegaMenu = ({ onAdminToggle }) => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

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
    { id: 'predict', name: 'Price Predictor', icon: '💰', path: '/' },
    { id: 'esg', name: 'ESG Agent', icon: '🌱', path: '/esg-agent' },
    { id: 'training', name: 'Model Training', icon: '�', path: '/training' },
    { id: 'finance', name: 'Financial Insights', icon: '📊', path: '/agent/finance' },
    { id: 'passive', name: 'Investment Analysis', icon: '�', path: '/agent/passive' }
  ];

  const handleItemClick = (item) => {
    if (item.action === 'admin') {
      onAdminToggle();
    }
    setIsOpen(false);
  };

  return (
    <>
      {/* Main Navigation Bar */}
      <nav className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg relative z-30">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            {/* Logo/Brand */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
                🏡
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
                        location.pathname === item.path ? 'bg-white bg-opacity-20' : ''
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
                  
                  {/* Hover Description */}
                  <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 px-3 py-1 bg-black bg-opacity-75 text-white text-sm rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50">
                    {item.description}
                  </div>
                </div>
              ))}

              {/* Real Estate Agents Dropdown */}
              <div className="relative group">
                <button className="px-4 py-2 rounded-lg transition-all duration-200 hover:bg-white hover:bg-opacity-20 flex items-center space-x-1">
                  <span>🏡 RE Agents</span>
                  <span className="text-xs">▼</span>
                </button>
                
                {/* Dropdown Menu */}
                <div className="absolute top-full left-0 mt-2 w-64 bg-white rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-50">
                  <div className="p-2">
                    <div className="text-xs text-gray-500 mb-2 px-2">REAL ESTATE AI PLATFORM</div>
                    {relevantAgents.map(agent => (
                      <Link
                        key={agent.id}
                        to={agent.path}
                        className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 transition-colors text-gray-800"
                      >
                        <span className="font-medium text-sm">{agent.name}</span>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Quick Actions */}
              <div className="flex items-center space-x-2 border-l border-white border-opacity-30 pl-4 ml-2">
                <button
                  onClick={onAdminToggle}
                  className="p-2 hover:bg-white hover:bg-opacity-20 rounded-lg transition-all duration-200 group"
                  title="Admin Panel (Ctrl+A)"
                >
                  <span className="text-lg group-hover:scale-110 transition-transform">⚙️</span>
                </button>
                <div className="w-8 h-8 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                  👤
                </div>
              </div>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 hover:bg-white hover:bg-opacity-20 rounded-lg transition-all duration-200"
            >
              <span className="text-xl">{isOpen ? '✕' : '☰'}</span>
            </button>
          </div>
        </div>

        {/* Mobile Mega Menu */}
        {isOpen && (
          <div className="md:hidden bg-white text-gray-800 shadow-lg">
            <div className="px-4 py-2 space-y-1">
              {/* Main Menu Items */}
              {menuItems.map(item => (
                <div key={item.id} className="border-b border-gray-100 last:border-b-0 py-3">
                  {item.path ? (
                    <Link
                      to={item.path}
                      onClick={() => setIsOpen(false)}
                      className={`flex items-center space-x-3 p-3 rounded-lg transition-colors ${
                        location.pathname === item.path 
                          ? 'bg-blue-50 text-blue-600' 
                          : 'hover:bg-gray-50'
                      }`}
                    >
                      <span className="text-xl">{item.label.split(' ')[0]}</span>
                      <div>
                        <div className="font-medium">{item.label.substring(2)}</div>
                        <div className="text-sm text-gray-500">{item.description}</div>
                      </div>
                    </Link>
                  ) : (
                    <button
                      onClick={() => handleItemClick(item)}
                      className="w-full flex items-center space-x-3 p-3 rounded-lg transition-colors hover:bg-gray-50"
                    >
                      <span className="text-xl">{item.label.split(' ')[0]}</span>
                      <div className="text-left">
                        <div className="font-medium">{item.label.substring(2)}</div>
                        <div className="text-sm text-gray-500">{item.description}</div>
                      </div>
                    </button>
                  )}
                </div>
              ))}
              
              {/* Real Estate Agents Section */}
              <div className="py-3 border-t border-gray-200">
                <div className="text-sm font-medium text-gray-500 mb-2 px-3">🏡 REAL ESTATE AI</div>
                {relevantAgents.map(agent => (
                  <Link
                    key={agent.id}
                    to={agent.path}
                    onClick={() => setIsOpen(false)}
                    className="flex items-center space-x-3 p-3 rounded-lg transition-colors hover:bg-gray-50"
                  >
                    <div className="font-medium text-sm">{agent.name}</div>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Desktop Mega Menu (on hover) */}
      <div className="hidden md:block absolute top-16 left-0 right-0 bg-white shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-20">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="grid grid-cols-3 gap-6">
            {menuItems.map(item => (
              <div key={item.id} className="group cursor-pointer">
                {item.path ? (
                  <Link to={item.path} className="block p-4 rounded-lg hover:bg-gray-50 transition-colors">
                    <div className="text-lg font-medium text-gray-800 mb-2">{item.label}</div>
                    <div className="text-sm text-gray-600">{item.description}</div>
                  </Link>
                ) : (
                  <button
                    onClick={() => handleItemClick(item)}
                    className="w-full text-left block p-4 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <div className="text-lg font-medium text-gray-800 mb-2">{item.label}</div>
                    <div className="text-sm text-gray-600">{item.description}</div>
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Access Floating Hints */}
      <div className="fixed bottom-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded-lg text-sm z-40">
        💡 <kbd className="bg-white bg-opacity-20 px-1 rounded">Ctrl+A</kbd> for Admin Panel
      </div>
    </>
  );
};

export default GlobalMegaMenu;
