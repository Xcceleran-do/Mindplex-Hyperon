import React, {useState} from 'react'

export default function RecommendationsPanel({recommendations=[]}){
  return (
    <div style={{marginTop:20}}>
      <h3>Recommendations</h3>
      <ul>
        {recommendations.map(r=> (
          <li key={r.contentId}>
            <div><strong>{r.title}</strong> — score: {r.score}</div>
          </li>
        ))}
      </ul>
    </div>
  )
}
