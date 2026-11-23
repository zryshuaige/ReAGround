import { useEffect, useState } from 'react'
import { api, errorMessage } from './api'
import ChatPage from './components/ChatPage'
import EvaluationPage from './components/EvaluationPage'
import Sidebar from './components/Sidebar'
import type { ModelInfo } from './types'
import styles from './App.module.css'

type Page = 'demo' | 'evaluation'

export default function App() {
  const [page, setPage] = useState<Page>('demo')
  const [collapsed, setCollapsed] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState('original')
  const [error, setError] = useState('')

  useEffect(() => {
    api.getModels().then(({ models: rows }) => {
      setModels(rows)
      const preferred = rows.find(row => row.id === 'direct_a' && row.available)
        ?? rows.find(row => row.available)
      if (preferred) setSelectedModel(preferred.id)
    }).catch(error => setError(errorMessage(error)))
  }, [])

  return (
    <div className={styles.app}>
      <Sidebar page={page} collapsed={collapsed} onCollapse={() => setCollapsed(value => !value)} onNavigate={setPage} />
      <main className={styles.main}>
        {error && <div className={styles.globalError}>{error}</div>}
        <div className={styles.page} hidden={page !== 'demo'}>
          <ChatPage models={models} selectedModel={selectedModel} onModelChange={setSelectedModel} />
        </div>
        {page === 'evaluation' && <EvaluationPage />}
      </main>
    </div>
  )
}
