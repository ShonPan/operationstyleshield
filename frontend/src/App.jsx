import { useState } from 'react'
import './App.css'

const COLS = [
  { key: 'account_id', label: 'account_id' },
  { key: 'confidence', label: 'confidence' },
  { key: 'bot_score', label: 'bot_score' },
  { key: 'cluster_id', label: 'cluster_id' },
  { key: 'vocabulary_uniformity', label: 'vocabulary_uniformity' },
  { key: 'structural_regularity', label: 'structural_regularity' },
  { key: 'hedging_signature', label: 'hedging_signature' },
]

function formatCell(key, val) {
  if (val == null) return '—'
  if (typeof val === 'number' && key !== 'cluster_id') {
    return Number(val).toFixed(4)
  }
  return String(val)
}

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [log, setLog] = useState('')
  const [top15, setTop15] = useState([])
  const [meta, setMeta] = useState(null)

  async function runAnalyze() {
    if (!file) {
      setError('Choose a CSV file first.')
      return
    }
    setLoading(true)
    setError(null)
    setLog('')
    setTop15([])
    setMeta(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: fd,
      })
      const text = await res.text()
      let data
      try {
        data = JSON.parse(text)
      } catch {
        setError(
          `Bad response (HTTP ${res.status}). Start API from project folder:\n` +
            `cd api && uvicorn main:app --reload --port 8000\n\n` +
            (text.slice(0, 280) || '(empty body)'),
        )
        return
      }
      if (!data.ok) {
        const parts = [
          data.error,
          typeof data.detail === 'string' && data.detail !== data.error ? data.detail : null,
        ].filter(Boolean)
        if (!parts.length && Array.isArray(data.detail)) {
          parts.push(data.detail.map((d) => d.msg || JSON.stringify(d)).join('; '))
        }
        setError(parts.join('\n\n---\n') || `HTTP ${res.status} — Request failed`)
        return
      }
      setTop15(data.top15 || [])
      setLog(data.log || '')
      setMeta({
        total_accounts: data.total_accounts,
        clusters: data.clusters,
      })
    } catch (e) {
      setError(e.message || 'Network error — is the API running? (uvicorn api.main:app)')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">◆</span>
          <div>
            <h1>StyleShield</h1>
            <p className="tagline">Upload CSV → Python pipeline → top accounts by confidence</p>
          </div>
        </div>
      </header>

      <section className="panel upload-panel">
        <label className="file-label">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <span className="file-btn">{file ? file.name : 'Choose CSV…'}</span>
        </label>
        <button
          type="button"
          className="run-btn"
          disabled={loading || !file}
          onClick={runAnalyze}
        >
          {loading ? 'Running Python…' : 'Run StyleShield'}
        </button>
      </section>

      {error && <div className="alert error">{error}</div>}

      {meta && (
        <section className="panel meta">
          <strong>{meta.total_accounts}</strong> accounts analyzed
          {meta.clusters && Object.keys(meta.clusters).length > 0 && (
            <span className="muted">
              {' · '}
              {Object.keys(meta.clusters).length} cluster(s)
            </span>
          )}
        </section>
      )}

      {top15.length > 0 && (
        <section className="panel results">
          <h2>TOP 15 ACCOUNTS BY CONFIDENCE:</h2>
          <div className="rule" />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {COLS.map((c) => (
                    <th key={c.key}>{c.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {top15.map((row, i) => (
                  <tr key={i}>
                    {COLS.map((c) => (
                      <td key={c.key} className={c.key === 'account_id' ? 'mono' : ''}>
                        {formatCell(c.key, row[c.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {log && (
        <section className="panel log-panel">
          <h3>Pipeline log</h3>
          <pre className="log">{log}</pre>
        </section>
      )}

      <footer className="footer">
        <p>
          API (required):{' '}
          <code>
            cd api → pip install -r requirements.txt → pip install numpy pandas scikit-learn → uvicorn main:app
            --reload --port 8000
          </code>
        </p>
      </footer>
    </div>
  )
}
