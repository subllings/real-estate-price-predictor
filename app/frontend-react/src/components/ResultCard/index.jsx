// src/components/ResultCard/index.jsx
import React from "react";


const ResultCard = ({ title, value }) => {
  return (
    <div className="bg-green-100 border-l-4 border-green-500 p-4 shadow rounded">
      <h3 className="text-lg font-bold text-green-700">{title}</h3>
      <p className="text-2xl mt-2 font-semibold">
        Estimated Price (€): {parseInt(value).toLocaleString("fr-FR")}
      </p>
    </div>
  );
};

export default ResultCard;
