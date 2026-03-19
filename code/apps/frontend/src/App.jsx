import { useState } from "react";
import "./App.css";

export default function App() {
  const [company, setCompany] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company }),
      });

      const data = await res.json();
      console.log("Model result:", data);
      setResult(data);
    } catch (err) {
      console.error("Error calling model:", err);
      setResult(null);
    }

    setLoading(false);
  }

  return (
    <div className="page">
      <div className="card">
        <header className="header">
          <h1>Company Classifier</h1>
          <p>
            Enter a company name to predict its <strong>industry</strong>,{" "}
            <strong>group</strong>, and <strong>business</strong>.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="form">
          <input
            type="text"
            placeholder="Enter a company name..."
            value={company}
            onChange={(e) => setCompany(e.target.value)}
          />

          <button type="submit" disabled={loading || company.length === 0}>
            {loading ? "Running Model..." : "Predict"}
          </button>
        </form>

        {result && (
          <>
            {/* Predictions */}
            <div className="results">
              <ResultCard title="Industry" value={result.industry} />
              <ResultCard title="Group" value={result.group} />
              <ResultCard title="Business" value={result.business} />
            </div>

            {/* Confidence */}
            <div className="confidence">
              Model confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            </div>

            {/* Similar organizations */}
            <div className="similar-section">
              <h3>Closest Organizations</h3>

              <div className="results">
                <ResultCard
                  title="Closest"
                  value={result.similar_orgs[0]?.name || "—"}
                />
                <ResultCard
                  title="Second Closest"
                  value={result.similar_orgs[1]?.name || "—"}
                />
                <ResultCard
                  title="Third Closest"
                  value={result.similar_orgs[2]?.name || "—"}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ResultCard({ title, value }) {
  return (
    <div className="result-card">
      <span className="result-label">{title}</span>
      <span className="result-value">{value}</span>
    </div>
  );
}