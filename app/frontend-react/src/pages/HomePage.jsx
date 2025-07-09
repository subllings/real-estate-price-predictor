import React from "react";
import AgentCard from "../components/AgentCard/AgentCard.jsx";

const agents = [
  {
    id: "esg",
    name: "ESG Agent",
    description: "Analyze ESG data (CO₂, energy...), auto-generate CSRD reports",
    image: "/images/esg256x256.png",
  },
  {
    id: "software",
    name: "Software Engineering Agent",
    description: "Assist in code analysis and debugging",
    image: "/images/softeng256x256.png",
  },
  {
    id: "ecommerce",
    name: "E-commerce Analytics Agent",
    description: "Analyze top-selling products, AI-powered recommendations",
    image: "/images/ebusiness256x256.png",
  },
  {
    id: "finance",
    name: "Financial Insights Agent",
    description: "Investment decisions",
    image: "/images/investment256x256.png",
  },
  {
    id: "passive",
    name: "Passive Income Agent",
    description: "Generate passive income ideas with AI evaluation",
    image: "/images/passiveincome256x256.png",
  },
  {
    id: "claims",
    name: "Claims Automation Agent",
    description: "Automate insurance claims processing",
    image: "/images/claims256x256.png",
  },
];

export default function HomePage() {
  return (
    <div className="px-6 py-10">
      <h1 className="text-3xl font-bold mb-10 text-center">Welcome to NeuroMesh</h1>
      <p className="text-gray-500 text-lg mt-2 mb-8 text-center">
        Select an intelligent agent to begin:
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mt-10">
        {agents.map((agent) => (
          <AgentCard
            key={agent.id}
            title={agent.name}
            imageSrc={agent.image}
            description={agent.description}
            path={`/agent/${agent.id}`}
          />
        ))}
      </div>
    </div>
  );
}
