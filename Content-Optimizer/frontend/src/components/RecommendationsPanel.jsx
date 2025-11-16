import { useState, useEffect, useCallback } from 'react'
import React from 'react'
const API_BASE = 'http://127.0.0.1:8000'

export function RecommendationsPanel() {
  const [creatorId, setCreatorId] = useState('')
  const [topK, setTopK] = useState(5)
  const [recs, setRecs] = useState([])
  const [modelVersion, setModelVersion] = useState(null)
  const [strategy, setStrategy] = useState({ items: [], summary: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [viewMode, setViewMode] = useState('list') // 'list' | 'grid'
  const [expandedId, setExpandedId] = useState(null)
  const [sortDir, setSortDir] = useState('desc') // 'asc'|'desc'

  // Dark mode toggle (adds/removes 'dark' class on <html>)
  const toggleDark = () => {
    const root = document.documentElement
    root.classList.toggle('dark')
  }

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      const data = await res.json()
      setHealth(data)
    } catch {
      // ignore health errors here
    }
  }, [])

  const fetchRecs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const url = new URL(`${API_BASE}/recommendations`)
      if (creatorId) url.searchParams.set('creatorId', creatorId)
      url.searchParams.set('topK', topK)
      const res = await fetch(url.toString())
      if (!res.ok) {
        throw new Error(`Request failed ${res.status}`)
      }
      const data = await res.json()
      let items = data.recommendations || []
      // apply client-side sort direction toggle
      items = [...items].sort((a,b) => sortDir === 'desc' ? b.score - a.score : a.score - b.score)
      setRecs(items)
      setModelVersion(data.modelVersion || null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [creatorId, topK, sortDir])

  const fetchStrategy = useCallback(async () => {
    try {
      const url = new URL(`${API_BASE}/strategy`)
      if (creatorId) url.searchParams.set('creatorId', creatorId)
      url.searchParams.set('topK', 3)
      const res = await fetch(url.toString())
      if (!res.ok) return
      const data = await res.json()
      setStrategy(data)
    } catch {
      // ignore
    }
  }, [creatorId])

  const refreshCache = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/recommendations/refresh`)
      if (!res.ok) throw new Error('Refresh failed')
      await fetchRecs()
    } catch (e) {
      setError(e.message)
      setLoading(false)
    }
  }, [fetchRecs])

  useEffect(() => {
    fetchHealth()
    fetchRecs()
    fetchStrategy()
  }, [fetchHealth, fetchRecs, fetchStrategy])

  return (
    <div className="w-full max-w-4xl mx-auto relative group bg-white/70 dark:bg-slate-900/70 backdrop-blur border border-slate-200 dark:border-slate-700 shadow-md rounded-xl p-6 space-y-6 overflow-hidden">
      <div className="absolute inset-0 -z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-700">
        <div className="pointer-events-none absolute -inset-[2px] rounded-xl bg-linear-to-r from-brand-400 via-pink-400 to-violet-400 blur-xl opacity-40" />
      </div>
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-800 dark:text-slate-100 flex items-center gap-2">
            <span>Recommendations</span>
            {modelVersion && (
              <span className="text-xs rounded-full bg-brand-600/10 text-brand-700 dark:text-brand-300 px-2 py-1 border border-brand-600/20">{modelVersion}</span>
            )}
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">Hybrid scoring from graph + text embeddings</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button onClick={fetchRecs} disabled={loading} className="px-3 py-2 text-sm font-medium rounded-md bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-50 shadow-sm">Load</button>
          <button onClick={refreshCache} disabled={loading} className="px-3 py-2 text-sm font-medium rounded-md bg-slate-700 text-white hover:bg-slate-600 disabled:opacity-50 shadow-sm">Refresh</button>
          <button onClick={() => setViewMode(viewMode === 'list' ? 'grid' : 'list')} className="px-3 py-2 text-sm font-medium rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 shadow-sm">{viewMode === 'list' ? 'Grid' : 'List'} View</button>
          <button onClick={() => setSortDir(sortDir === 'desc' ? 'asc' : 'desc')} className="px-3 py-2 text-sm font-medium rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 shadow-sm">Sort: {sortDir}</button>
          <button onClick={toggleDark} className="px-3 py-2 text-sm font-medium rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 shadow-sm">Theme</button>
        </div>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); fetchRecs(); }} className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium uppercase tracking-wide text-slate-500">Creator ID</label>
          <input
            className="px-3 py-2 rounded-md border border-slate-300 bg-white dark:bg-slate-800 dark:border-slate-600 focus:outline-none focus:ring-2 focus:ring-brand-500"
            value={creatorId}
            onChange={(e) => setCreatorId(e.target.value)}
            placeholder="creator_1"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium uppercase tracking-wide text-slate-500">Top K</label>
          <input
            type="number"
            min={1}
            max={50}
            className="px-3 py-2 rounded-md border border-slate-300 bg-white dark:bg-slate-800 dark:border-slate-600 focus:outline-none focus:ring-2 focus:ring-brand-500"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value) || 1)}
          />
        </div>
        <div className="flex flex-col justify-end">
          <button type="submit" className="px-3 py-2 rounded-md bg-brand-600 text-white text-sm font-medium hover:bg-brand-500">Apply</button>
        </div>
      </form>

      {health && (
        <div className="text-xs text-slate-600 dark:text-slate-400 flex flex-wrap gap-3">
          <span>Neo4j: {health.neo4j_version || 'n/a'}</span>
          <span>Content: {health.content_nodes ?? 'n/a'}</span>
          <span>Embeddings: {health.has_embeddings ? 'yes' : 'no'}</span>
        </div>
      )}

      {loading && (
        <div className="space-y-2">
          {Array.from({ length: Math.min(topK, 6) }).map((_, i) => (
            <div key={i} className="animate-pulse h-14 rounded-md bg-slate-200/60 dark:bg-slate-700/50" />
          ))}
        </div>
      )}
      {error && <p className="text-sm text-red-600">Error: {error}</p>}

      {!loading && !error && (
        <div>
          {recs.length === 0 && <p className="py-4 text-sm text-slate-500">No recommendations</p>}
          {viewMode === 'list' && recs.length > 0 && (
            <ul className="divide-y divide-slate-200 dark:divide-slate-700">
              {recs.map(r => {
                const scoreClass = r.score >= 0.75 ? 'bg-green-500/20 text-green-800 dark:text-green-200' : r.score >= 0.4 ? 'bg-yellow-500/20 text-yellow-800 dark:text-yellow-200' : 'bg-red-500/20 text-red-800 dark:text-red-200'
                return (
                  <li key={r.contentId} className="py-3 flex items-center justify-between group">
                    <div className="flex flex-col">
                      <button type="button" onClick={() => setExpandedId(expandedId === r.contentId ? null : r.contentId)} className="text-left font-medium text-slate-800 dark:text-slate-100 hover:underline">
                        {r.title || r.contentId}
                      </button>
                      <span className="text-xs text-slate-500 dark:text-slate-400">{r.contentId}</span>
                      {expandedId === r.contentId && (
                        <div className="mt-1 text-xs text-slate-600 dark:text-slate-300 space-y-1">
                          <p>Model: {r.explanation?.model || 'n/a'}</p>
                          <p>Raw score: {r.score.toFixed(6)}</p>
                        </div>
                      )}
                    </div>
                    <div className="flex flex-col items-end text-xs text-slate-600 dark:text-slate-400">
                      <span className={`font-mono text-sm px-2 py-1 rounded-md ${scoreClass}`}>{r.score.toFixed(4)}</span>
                      {r.explanation?.model && <span className="opacity-60">{r.explanation.model}</span>}
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
          {viewMode === 'grid' && recs.length > 0 && (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {recs.map(r => {
                const scoreClass = r.score >= 0.75 ? 'bg-green-500/20 text-green-800 dark:text-green-200' : r.score >= 0.4 ? 'bg-yellow-500/20 text-yellow-800 dark:text-yellow-200' : 'bg-red-500/20 text-red-800 dark:text-red-200'
                return (
                  <div key={r.contentId} className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
                    <div className="space-y-1">
                      <h3 className="font-semibold text-slate-800 dark:text-slate-100 line-clamp-2">{r.title || r.contentId}</h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400">{r.contentId}</p>
                    </div>
                    <div className="mt-2 flex items-center justify-between">
                      <span className={`font-mono text-sm px-2 py-1 rounded-md ${scoreClass}`}>{r.score.toFixed(4)}</span>
                      {r.explanation?.model && <span className="text-[10px] text-slate-500 dark:text-slate-400">{r.explanation.model}</span>}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
      {/* Strategy section */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Strategy suggestions</h3>
        {strategy?.items?.length ? (
          <div className="mt-3 grid sm:grid-cols-2 gap-3">
            {strategy.items.map((s, idx) => (
              <div key={idx} className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm text-slate-500 dark:text-slate-400">Platform</div>
                    <div className="font-medium text-slate-800 dark:text-slate-100">{s.platform || 'n/a'}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-slate-500 dark:text-slate-400">Format</div>
                    <div className="font-medium text-slate-800 dark:text-slate-100">{s.format || 'n/a'}</div>
                  </div>
                </div>
                <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                  {s.lengthRangeSec ? (
                    <span>Optimal length: {Math.round(s.lengthRangeSec[0]/60)}–{Math.round(s.lengthRangeSec[1]/60)} min ({s.lengthRangeSec[0]}–{s.lengthRangeSec[1]}s)</span>
                  ) : (
                    <span>Optimal length: n/a</span>
                  )}
                </div>
                {s.examples?.length ? (
                  <div className="mt-2 text-xs text-slate-500">Examples: {s.examples.join(', ')}</div>
                ) : null}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500 mt-2">No strategy suggestions available.</p>
        )}
        {strategy?.summary && <p className="text-xs text-slate-500 mt-2">{strategy.summary}</p>}
      </div>
      {modelVersion && <p className="text-xs text-slate-500">Model version: <span className="font-semibold">{modelVersion}</span></p>}
    </div>
  )
}

export default RecommendationsPanel
