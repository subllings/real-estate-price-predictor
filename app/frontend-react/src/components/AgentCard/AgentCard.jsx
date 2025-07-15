/**
 * Agent Card Component - For multi-agent platform
 */

import React from 'react';
import { Link } from 'react-router-dom';

const AgentCard = ({ title, imageSrc, description, path }) => {
  return (
    <Link to={path} className="block">
      <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-300 overflow-hidden group">
        {/* Image Section */}
        <div className="h-48 bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center group-hover:from-blue-100 group-hover:to-purple-100 transition-all duration-300">
          {imageSrc && imageSrc.startsWith('/images/') ? (
            <img 
              src={imageSrc} 
              alt={title}
              className="w-24 h-24 object-contain group-hover:scale-110 transition-transform duration-300"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
          ) : null}
          {/* Fallback icon */}
          <div className="w-24 h-24 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white text-3xl group-hover:scale-110 transition-transform duration-300" style={{display: imageSrc && imageSrc.startsWith('/images/') ? 'none' : 'flex'}}>
            {getAgentIcon(title)}
          </div>
        </div>

        {/* Content Section */}
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2 group-hover:text-blue-600 transition-colors">
            {title}
          </h3>
          <p className="text-gray-600 text-sm leading-relaxed">
            {description}
          </p>
          
          {/* Action indicator */}
          <div className="mt-4 flex items-center text-blue-600 text-sm font-medium">
            <span>Explore Agent</span>
            <span className="ml-1 group-hover:translate-x-1 transition-transform">→</span>
          </div>
        </div>

        {/* Status indicator */}
        <div className="absolute top-3 right-3">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
        </div>
      </div>
    </Link>
  );
};

// Helper function to get appropriate icon for each agent
const getAgentIcon = (title) => {
  const icons = {
    'ESG Agent': '🌱',
    'Software Engineering Agent': '💻',
    'E-commerce Analytics Agent': '🛒',
    'Financial Insights Agent': '📊',
    'Passive Income Agent': '💸',
    'Claims Automation Agent': '📋'
  };
  
  return icons[title] || '🤖';
};

export default AgentCard;
