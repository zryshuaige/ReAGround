import { useEffect, useRef, useState } from 'react'
import { CheckIcon, ChevronDown, SettingsIcon } from './Icons'
import styles from './InferenceSettings.module.css'

interface Props {
  visualTokens: number
  disabled?: boolean
  onChange(value: number): void
}

export default function InferenceSettings({ visualTokens, disabled = false, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', escape)
    }
  }, [open])

  return <div className={styles.root} ref={root}>
    <button type="button" className={styles.trigger} disabled={disabled}
      aria-haspopup="dialog" aria-expanded={open} aria-label="推理设置"
      onClick={() => setOpen(value => !value)}>
      <SettingsIcon size={15} />
      <span>SDPA · {visualTokens} tokens</span>
      <ChevronDown size={12} className={open ? styles.chevronOpen : ''} />
    </button>
    {open && <div className={styles.popover} role="dialog" aria-label="推理设置">
      <div className={styles.title}>推理设置</div>
      <div className={styles.row}>
        <span>推理加速</span>
        <span className={styles.selected}><CheckIcon size={13} />SDPA</span>
      </div>
      <div className={styles.tokenHeading}>
        <label htmlFor="visual-token-limit">图片 token 上限</label>
        <strong>{visualTokens}</strong>
      </div>
      <input id="visual-token-limit" className={styles.slider} type="range"
        min={256} max={2048} step={128} value={visualTokens} disabled={disabled}
        aria-valuetext={`${visualTokens} 视觉 tokens`}
        onChange={event => onChange(Number(event.target.value))} />
      <div className={styles.scale}><span>256</span><span>默认 1280</span><span>2048</span></div>
      <p>降低 token 可减少计算，但可能损害高分辨率小目标的定位准确率。</p>
    </div>}
  </div>
}
