import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const PredictionChart = ({ data }) => {
  if (!data) return null;

  // Construction des données à partir des valeurs numériques du formulaire
  const chartData = Object.entries(data)
    .filter(([key, value]) =>
      key !== 'ocean_proximity' && !isNaN(parseFloat(value))
    )
    .map(([key, value]) => ({
      name: key.replace(/_/g, ' '),
      value: parseFloat(value),
    }));

  return (
    <div style={{ width: '100%', height: 300, marginTop: 30 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#007bff" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictionChart;
