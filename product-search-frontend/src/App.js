import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults([]);
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) throw new Error('API error');
      const data = await response.json();
      setResults(data || []);
    } catch (err) {
      setError('Failed to fetch results.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Product Search</h1>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search for a product..."
            required
            className="search-input"
          />
          <button type="submit" className="search-btn" disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
        <div className="results">
          {results.length > 0 ? (
            <div className="card-list">
              {results.map((item, idx) => (
                <div className="product-card" key={idx}>
                  <div className="product-header">
                    <h2 className="product-title">{item.title}</h2>
                    {item.is_bestseller && <span className="product-badge bestseller">Bestseller</span>}
                    {item.is_featured && <span className="product-badge featured">Featured</span>}
                  </div>
                  
                  <div className="product-details">
                    <div className="product-info">
                      <p className="product-brand">Brand: <span>{item.brand}</span></p>
                      <p className="product-category">Category: <span>{item.category}</span></p>
                      {item.subcategory && <p className="product-subcategory">Subcategory: <span>{item.subcategory}</span></p>}
                      {item.color && <p className="product-color">Color: <span>{item.color}</span></p>}
                      {item.material && <p className="product-material">Material: <span>{item.material}</span></p>}
                      {item.size_range && <p className="product-size">Size: <span>{item.size_range}</span></p>}
                      {item.gender && <p className="product-gender">Gender: <span>{item.gender}</span></p>}
                    </div>
                    
                    <div className="product-description">{item.description}</div>
                    
                    <div className="product-meta">
                      <div className="product-price-container">
                        <div className="product-price">${item.price}</div>
                        {item.discount > 0 && (
                          <div className="product-discount">{item.discount}% off</div>
                        )}
                      </div>
                      
                      <div className="product-rating">
                        <span className="stars">{'★'.repeat(Math.floor(item.rating))}{'☆'.repeat(5-Math.floor(item.rating))}</span>
                        <span className="rating-value">{item.rating}</span>
                      </div>
                    </div>
                    
                    <div className="product-inventory">
                      {/* <p className="product-sku">SKU: {item.sku}</p> */}
                      <p className="product-stock">In Stock: {item.stock}</p>
                    </div>
                    
                    {item._nlp_score && (
                      <div className="nlp-score">
                        <span className="nlp-score-label">NLP Score:</span>
                        <span className="nlp-score-value">{item._nlp_score.toFixed(2)}</span>
                      </div>
                    )}
                    
                    {item.tags && item.tags.length > 0 && (
                      <div className="product-tags">
                        {item.tags.map((tag, tagIdx) => (
                          <span className="tag" key={tagIdx}>{tag}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            !loading && <div>No results to display.</div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
