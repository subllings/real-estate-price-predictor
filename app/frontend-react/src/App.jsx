import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import RealEstatePredictorPage from './pages/RealEstatePredictorPage';
import ESGAgentPage from './pages/ESGAgentPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<RealEstatePredictorPage />} />
        <Route path="/esg-agent" element={<ESGAgentPage />} />
      </Routes>
    </Router>
  );
}

export default App;
