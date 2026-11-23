import { useEffect, useRef, useState } from 'react'
import type { ModelInfo } from '../types'
import { CheckIcon, ChevronDown, ModelIcon } from './Icons'
import styles from './ModelSelect.module.css'

interface Props {
  models: ModelInfo[]
  value: string
  disabled?: boolean
  onChange(id: string): void
}

export default function ModelSelect({ models, value, disabled = false, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const current = models.find(model => model.id === value)

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

  return (
    <div className={styles.root} ref={root}>
      <button type="button" className={styles.trigger} disabled={disabled} aria-haspopup="menu" aria-expanded={open} aria-label={`当前模型：${current?.name ?? '加载中'}`} onClick={() => setOpen(value => !value)} onKeyDown={event => {
        if (event.key === 'ArrowDown') {
          event.preventDefault()
          setOpen(true)
        }
      }}>
        <ModelIcon size={16} />
        <span className={styles.label}>{current?.name ?? '加载模型中…'}</span>
        <ChevronDown size={13} className={open ? styles.chevronOpen : ''} />
      </button>
      {open && <div className={styles.menu} role="menu" aria-label="选择模型">
        <div className={styles.menuCaption}>选择本地模型</div>
        {models.map(model => <button
          type="button"
          role="menuitemradio"
          aria-checked={model.id === value}
          className={`${styles.option} ${model.id === value ? styles.selected : ''}`}
          key={model.id}
          disabled={!model.available}
          onClick={() => {
            onChange(model.id)
            setOpen(false)
          }}
        >
          <span className={styles.optionText}>{model.name}{!model.available && <small>权重未就绪</small>}</span>
          <span className={styles.check}>{model.id === value && <CheckIcon size={15} />}</span>
        </button>)}
      </div>}
    </div>
  )
}
