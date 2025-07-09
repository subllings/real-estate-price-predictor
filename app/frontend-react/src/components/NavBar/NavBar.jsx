import React from "react";
import { Link } from "react-router-dom";

export default function NavBar() {
  return (
    <nav style={{ padding: "1rem", background: "#f5f5f5" }}>
      <Link to="/" style={{ marginRight: "1rem" }}>Home</Link>
      <Link to="/agent/esg">ESG Agent</Link>
    </nav>
  );
}
