import React from 'react'
import './App.css'
import RecommendationsPanel from './components/RecommendationsPanel.jsx'

function App() {
  return (
  <div className="min-h-screen bg-linear-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      <div className="mx-auto max-w-5xl px-4 py-10 space-y-10">
        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-brand-600 to-brand-400 dark:from-brand-400 dark:to-brand-200">
            Content Optimizer Demo
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Hybrid recommendations (GraphSAGE + text) served via FastAPI
          </p>
        </header>
        <RecommendationsPanel />
        <footer className="pt-6 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500 flex flex-wrap gap-4">
          <span>Frontend Step 10</span>
          <span>
            <a href="https://fastapi.tiangolo.com/" className="hover:text-brand-600" target="_blank" rel="noreferrer">FastAPI</a>
            {' '}•{' '}
            <a href="https://vite.dev" className="hover:text-brand-600" target="_blank" rel="noreferrer">Vite</a>
          </span>
          <span className="ml-auto">Dark mode supported</span>
        </footer>
      </div>
    </div>
  )
}

export default App

