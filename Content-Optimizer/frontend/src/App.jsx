import React, {useState} from 'react'
import axios from 'axios'
import RecommendationsPanel from './components/RecommendationsPanel'

export default function App(){
  const [creatorId, setCreatorId] = useState('')
  const [recs, setRecs] = useState([])

  const getRecs = async ()=>{
    try{
      const res = await axios.post('http://localhost:8000/recommendations', {creatorId})
      setRecs(res.data.recommendations || [])
    }catch(err){
      console.error(err)
    }
  }

  return (
    <div style={{padding:20}}>
      <h2>Content Optimizer</h2>
      <input value={creatorId} onChange={e=>setCreatorId(e.target.value)} placeholder="creatorId" />
      <button onClick={getRecs}>Get Recommendations</button>
      <RecommendationsPanel recommendations={recs} />
    </div>
  )
}
