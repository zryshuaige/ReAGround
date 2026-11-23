import { useEffect, useRef, useState } from 'react'
import { imageUrl } from '../api'
import { CloseIcon, ExpandIcon, MinusIcon, PlusIcon } from './Icons'
import styles from './ImageViewer.module.css'

export default function ImageViewer({ path, onClose }: { path: string; onClose(): void }) {
  const [natural, setNatural] = useState({ width: 1, height: 1 })
  const [scale, setScale] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null)

  const fit = () => Math.min(
    (window.innerWidth - 80) / natural.width,
    (window.innerHeight - 150) / natural.height,
    1,
  )

  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', key)
    return () => document.removeEventListener('keydown', key)
  }, [onClose])

  const zoom = (next: number) => {
    setScale(Math.max(0.1, Math.min(5, next)))
    setPan({ x: 0, y: 0 })
  }

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-label="预测截图全屏查看">
      <div className={styles.toolbar}>
        <strong>预测截图</strong>
        <div>
          <button onClick={() => zoom(fit())}><ExpandIcon size={15} />适应窗口</button>
          <button onClick={() => zoom(1)}>100%</button>
          <button onClick={() => zoom(scale - 0.2)} aria-label="缩小"><MinusIcon /></button>
          <span>{Math.round(scale * 100)}%</span>
          <button onClick={() => zoom(scale + 0.2)} aria-label="放大"><PlusIcon /></button>
          <button onClick={onClose} aria-label="关闭"><CloseIcon /></button>
        </div>
      </div>
      <div
        className={styles.stage}
        onWheel={event => {
          event.preventDefault()
          zoom(scale + (event.deltaY < 0 ? 0.12 : -0.12))
        }}
        onPointerDown={event => {
          drag.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y }
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={event => {
          if (!drag.current) return
          setPan({
            x: drag.current.panX + event.clientX - drag.current.x,
            y: drag.current.panY + event.clientY - drag.current.y,
          })
        }}
        onPointerUp={() => { drag.current = null }}
        onPointerCancel={() => { drag.current = null }}
      >
        <img
          src={imageUrl(path)}
          alt="全屏预测截图"
          draggable={false}
          onLoad={event => {
            const image = event.currentTarget
            const size = { width: image.naturalWidth, height: image.naturalHeight }
            setNatural(size)
            const initial = Math.min((window.innerWidth - 80) / size.width, (window.innerHeight - 150) / size.height, 1)
            setScale(initial)
          }}
          style={{
            width: natural.width * scale,
            height: natural.height * scale,
            transform: `translate(${pan.x}px, ${pan.y}px)`,
          }}
        />
      </div>
      <div className={styles.help}>滚轮缩放 · 拖动图片 · Esc 关闭</div>
    </div>
  )
}
