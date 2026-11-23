import { useEffect, useRef, useState } from 'react'
import { api, errorMessage, imageUrl } from '../api'
import type { CaptureResult, ModelInfo, PredictionResult } from '../types'
import ImageViewer from './ImageViewer'
import { ExpandIcon, SendIcon, WarningIcon } from './Icons'
import InferenceSettings from './InferenceSettings'
import ModelSelect from './ModelSelect'
import styles from './ChatPage.module.css'

interface TextItem {
  id: string
  kind: 'text'
  role: 'user' | 'assistant' | 'meta'
  text: string
}
interface ImageItem { id: string; kind: 'image'; path: string }
interface ConfirmItem { id: string; kind: 'confirm'; prediction: PredictionResult; status: 'pending' | 'cancelled' | 'clicked' }
type Item = TextItem | ImageItem | ConfirmItem

let sequence = 0
const nextId = () => `${Date.now()}-${sequence++}`

export default function ChatPage({ models, selectedModel, onModelChange }: {
  models: ModelInfo[]
  selectedModel: string
  onModelChange(id: string): void
}) {
  const [items, setItems] = useState<Item[]>([{
    id: 'welcome', kind: 'text', role: 'assistant',
    text: '你好，我是 ReAGround。告诉我你想点击的界面元素，我会截取当前桌面并标出预测位置。\n你可以放大检查截图，只有确认后才会执行点击。',
  }])
  const [input, setInput] = useState('')
  const [visualTokens, setVisualTokens] = useState(1280)
  const [busy, setBusy] = useState<'capture' | 'predict' | 'settle' | null>(null)
  const [viewer, setViewer] = useState<string | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const pending = items.some(item => item.kind === 'confirm' && item.status === 'pending')
  const locked = busy !== null || pending

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [items, busy])

  const append = (item: Item) => setItems(rows => [...rows, item])
  const appendText = (role: TextItem['role'], text: string) => append({ id: nextId(), kind: 'text', role, text })

  async function send() {
    const instruction = input.trim()
    if (!instruction || locked) return
    if (!models.some(model => model.id === selectedModel && model.available)) {
      appendText('assistant', '当前模型权重尚未配置，请先选择可用模型。')
      return
    }
    setInput('')
    appendText('user', instruction)
    setBusy('capture')
    try {
      const capture: CaptureResult = await api.capture()
      appendText('meta', `已截取 ${capture.width}×${capture.height} 桌面 · ${capture.timings.captureMs.toFixed(0)} ms`)
      setBusy('predict')
      const prediction = await api.predict({
        modelId: selectedModel, captureId: capture.captureId, instruction, visualTokens,
      })
      appendText('assistant', `模型输出\n${prediction.raw}\n\n截图坐标  (${prediction.x.toFixed(1)}, ${prediction.y.toFixed(1)})`)
      appendText('meta', `SDPA · 图片 token 上限 ${prediction.visualTokens} · 输入 ${prediction.resizedWh[0]}×${prediction.resizedWh[1]} · 推理 ${(prediction.timings.inferenceMs / 1000).toFixed(1)} s`)
      append({ id: nextId(), kind: 'image', path: prediction.imagePath })
      append({ id: nextId(), kind: 'confirm', prediction, status: 'pending' })
    } catch (error) {
      appendText('assistant', `运行失败：\n${errorMessage(error)}`)
    } finally {
      setBusy(null)
    }
  }

  async function settle(item: ConfirmItem, action: 'cancel' | 'click') {
    if (item.status !== 'pending' || busy) return
    setBusy('settle')
    try {
      if (action === 'cancel') await api.cancel(item.prediction.predictionId)
      else await api.click(item.prediction.predictionId)
      setItems(rows => rows.map(row => row.id === item.id
        ? { ...row, status: action === 'cancel' ? 'cancelled' : 'clicked' }
        : row))
    } catch (error) {
      appendText('assistant', `操作失败：\n${errorMessage(error)}`)
    } finally {
      setBusy(null)
    }
  }

  const status = busy === 'capture' ? '截取桌面中'
    : busy === 'predict' ? '模型推理中'
      : busy === 'settle' ? '处理操作中' : pending ? '等待确认' : '就绪'

  return (
    <section className={styles.page}>
      <header className={styles.header}>
        <div><h1>Demo</h1><p>用自然语言定位并点击桌面元素</p></div>
        <span className={styles.status}><i className={busy ? styles.busyDot : ''} />{status}</span>
      </header>
      <div className={styles.chat}>
        <div className={styles.chatContent}>
          {items.map(item => {
            if (item.kind === 'text') return <div key={item.id} className={`${styles.message} ${styles[item.role]}`}>
              <div className={styles.bubble}>{item.text}</div>
            </div>
            if (item.kind === 'image') return <button key={item.id} className={styles.imageCard} onClick={() => setViewer(item.path)}>
              <span className={styles.imageHeader}><strong>预测截图</strong><span><ExpandIcon size={14} /> 点击放大 · 全屏查看</span></span>
              <img src={imageUrl(item.path)} alt="带有预测点击位置的桌面截图" />
            </button>
            return <div key={item.id} className={styles.confirmCard}>
              <div className={styles.confirmTitle}><span><WarningIcon size={16} /></span><strong>准备执行点击</strong></div>
              <p>目标坐标 ({item.prediction.x.toFixed(1)}, {item.prediction.y.toFixed(1)})。请先检查上方截图，确认位置正确后再执行。此操作只会单击一次。</p>
              {item.status === 'pending'
                ? <div className={styles.confirmActions}>
                    <button onClick={() => settle(item, 'cancel')} disabled={busy !== null}>取消</button>
                    <button className={styles.primary} onClick={() => settle(item, 'click')} disabled={busy !== null}>确认点击</button>
                  </div>
                : <div className={styles.settled}>{item.status === 'cancelled' ? '已取消，未执行点击。' : '已执行一次鼠标左键单击。'}</div>}
            </div>
          })}
          {busy && <div className={styles.working}><span /><span /><span />{status}</div>}
          <div ref={bottom} />
        </div>
      </div>
      <div className={styles.composerWrap}>
        <div className={styles.composer}>
          <textarea
            value={input}
            readOnly={locked}
            placeholder="描述要点击的元素，例如：点击左上角的保存按钮…"
            onChange={event => setInput(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault()
                void send()
              }
            }}
          />
          <div className={styles.toolbar}>
            <ModelSelect models={models} value={selectedModel} onChange={onModelChange} disabled={locked} />
            <InferenceSettings visualTokens={visualTokens} onChange={setVisualTokens} disabled={locked} />
            <span className={styles.local}>本地桌面操作</span>
            <button className={styles.send} onClick={send} disabled={locked || !input.trim()} aria-label="发送" title="发送"><SendIcon size={18} /></button>
          </div>
        </div>
        <div className={styles.hint}>Enter 发送 · Shift+Enter 换行 · 点击预测图可全屏缩放</div>
      </div>
      {viewer && <ImageViewer path={viewer} onClose={() => setViewer(null)} />}
    </section>
  )
}
