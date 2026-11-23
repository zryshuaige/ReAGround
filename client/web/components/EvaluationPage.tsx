import { useEffect, useState } from 'react'
import { api, errorMessage } from '../api'
import type { BenchmarkMetrics, MetricValue, ResultEntry } from '../types'
import { ChartIcon } from './Icons'
import styles from './EvaluationPage.module.css'

const proGroups = [
  ['Dev', 'Development'], ['Creative', 'Creative'], ['CAD', 'CAD'],
  ['Scientific', 'Scientific'], ['Office', 'Office'], ['OS', 'Operating Systems'],
] as const

function percent(value?: number) {
  return value === undefined ? '—' : `${value.toFixed(2)}%`
}

function applications(metrics: BenchmarkMetrics) {
  return metrics.by_app ?? metrics.by_application ?? {}
}

function MetricCell({ value, overall = false }: { value?: MetricValue; overall?: boolean }) {
  return <td className={`${styles.metricCell} ${overall ? styles.overallCell : ''}`}>
    {value ? <><strong>{percent(value.accuracy_percent)}</strong><small>{value.hits} / {value.samples}</small></> : '—'}
  </td>
}

function AccuracyChart({ pro, v2 }: { pro: BenchmarkMetrics; v2: BenchmarkMetrics }) {
  const data = [
    { name: 'ScreenSpot-Pro', value: pro.accuracy_percent, hits: pro.hits, samples: pro.samples, color: '#536fe3', x: 205 },
    { name: 'ScreenSpot-v2', value: v2.accuracy_percent, hits: v2.hits, samples: v2.samples, color: '#42a58b', x: 465 },
  ]
  const baseline = 238
  const height = 176
  return (
    <div className={styles.chartCard}>
      <div className={styles.cardTitle}><div><h2>核心数据集准确率</h2><p>点落入 GT bbox 的样本比例</p></div><span>0–100%</span></div>
      <svg className={styles.chart} viewBox="0 0 680 330" role="img" aria-label={`ScreenSpot-Pro ${percent(pro.accuracy_percent)}，ScreenSpot-v2 ${percent(v2.accuracy_percent)}`}>
        {[0, 25, 50, 75, 100].map(tick => {
          const y = baseline - tick / 100 * height
          return <g key={tick}>
            <line x1="72" x2="640" y1={y} y2={y} stroke="#e9ecf1" strokeDasharray={tick === 0 ? undefined : '4 5'} />
            <text x="56" y={y + 4} textAnchor="end" fill="#9aa2af" fontSize="11">{tick}%</text>
          </g>
        })}
        {data.map(item => {
          const barHeight = item.value / 100 * height
          const y = baseline - barHeight
          return <g key={item.name}>
            <rect x={item.x} y={y} width="110" height={Math.max(barHeight, 1)} rx="8" fill={item.color} />
            <text x={item.x + 55} y={Math.max(y - 12, 20)} textAnchor="middle" fill="#303947" fontSize="18" fontWeight="600">{percent(item.value)}</text>
            <text x={item.x + 55} y="266" textAnchor="middle" fill="#4b5565" fontSize="13">{item.name}</text>
            <text x={item.x + 55} y="287" textAnchor="middle" fill="#9aa2af" fontSize="11">{item.hits} / {item.samples} 命中</text>
          </g>
        })}
      </svg>
    </div>
  )
}

export default function EvaluationPage() {
  const [results, setResults] = useState<ResultEntry[]>([])
  const [selected, setSelected] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    api.getResults().then(({ results: rows }) => {
      setResults(rows)
      const first = rows.find(row => row.modelId === 'direct_a' && row.metrics)
        ?? rows.find(row => row.metrics)
      if (first) setSelected(first.modelId)
    }).catch(error => setError(errorMessage(error))).finally(() => setLoading(false))
  }, [])

  const current = results.find(row => row.modelId === selected && row.metrics)
  const currentMetrics = current?.metrics

  return (
    <section className={styles.page}>
      <header className={styles.header}>
        <div><h1>模型评测</h1><p>ScreenSpot 系列 benchmark 的真实离线结果</p></div>
        <span className={styles.badge}><ChartIcon size={14} />本地结果</span>
      </header>
      <div className={styles.scroll}>
        <div className={styles.content}>
          <div className={styles.intro}>
            <div><h2>Grounding 表现概览</h2><p>图表、表格与指标卡均读取同一份 metrics.json，不包含预期值。</p></div>
            {results.some(row => row.metrics) && <label className={styles.modelFilter}>当前模型
              <select value={selected} onChange={event => setSelected(event.target.value)}>
                {results.filter(row => row.metrics).map(row => <option key={row.modelId} value={row.modelId}>{row.modelName}</option>)}
              </select>
            </label>}
          </div>
          {loading && <div className={styles.empty}>正在读取本地评测结果…</div>}
          {error && <div className={styles.error}>{error}</div>}
          {!loading && !error && results.length === 0 && <div className={styles.empty}><strong>尚未评测</strong><p>将模型的 metrics.json 配置到 client/models.json 后，这里会显示结果。</p></div>}
          {results.filter(row => row.error).map(row => <div className={styles.error} key={row.modelId}>{row.modelName}：{row.error}</div>)}
          {current && currentMetrics && <>
            <div className={styles.protocol}>
              <strong>正式评测输入条件</strong>
              <div>图片视觉 token 上限 <span>{current.visualTokens ?? '未记录'}</span> · 单图一次 Direct 推理 · 贪心解码</div>
              <small>Demo 的临时设置不会改写历史指标</small>
            </div>
            <div className={styles.overview}>
              <div className={styles.summaryCards}>
                <article className={styles.summaryCard}>
                  <span>ScreenSpot-Pro</span><strong>{percent(currentMetrics.screenspot_pro.accuracy_percent)}</strong>
                  <p>{currentMetrics.screenspot_pro.hits} / {currentMetrics.screenspot_pro.samples} 个样本命中{currentMetrics.screenspot_pro.parse_failures !== undefined && ` · ${currentMetrics.screenspot_pro.parse_failures} 条解析失败`}</p>
                  <i style={{ background: '#536fe3' }} />
                </article>
                <article className={styles.summaryCard}>
                  <span>ScreenSpot-v2</span><strong>{percent(currentMetrics.screenspot_v2.accuracy_percent)}</strong>
                  <p>{currentMetrics.screenspot_v2.hits} / {currentMetrics.screenspot_v2.samples} 个样本命中{currentMetrics.screenspot_v2.parse_failures !== undefined && ` · ${currentMetrics.screenspot_v2.parse_failures} 条解析失败`}</p>
                  <i style={{ background: '#42a58b' }} />
                </article>
                <article className={styles.modelCard}>
                  <span>当前模型</span><strong>{current.modelName}</strong>
                  <p>{currentMetrics.base_model ?? currentMetrics.model}</p>
                  <small>{currentMetrics.metric ?? '预测点落入 GT bbox 的样本比例'}</small>
                </article>
              </div>
              <AccuracyChart pro={currentMetrics.screenspot_pro} v2={currentMetrics.screenspot_v2} />
            </div>
          </>}
          {results.some(row => row.metrics?.screenspot_pro) && <>
            <div className={styles.sectionHeading}><h2>ScreenSpot-Pro 官方分组</h2><p>Overall 为全样本微平均，不是分组百分比的算术平均</p></div>
            <div className={styles.tableWrap}><table>
              <thead><tr><th>Model</th>{proGroups.map(([, label]) => <th key={label}>{label}</th>)}<th>Overall</th></tr></thead>
              <tbody>{results.filter(row => row.metrics?.screenspot_pro).map(row => {
                const metrics = row.metrics!.screenspot_pro
                return <tr key={row.modelId}>
                  <td className={styles.modelName}>{row.modelName}</td>
                  {proGroups.map(([key]) => <MetricCell key={key} value={metrics.by_official_group?.[key]} />)}
                  <MetricCell value={metrics} overall />
                </tr>
              })}</tbody>
            </table></div>
          </>}
          {results.some(row => row.metrics) && <>
            <div className={styles.sectionHeading}><h2>Text / Icon 表现</h2><p>按界面元素类型拆分</p></div>
            <div className={styles.tableWrap}><table>
              <thead><tr><th>Model</th><th>Benchmark</th><th>Text</th><th>Icon</th><th>Overall</th></tr></thead>
              <tbody>{results.filter(row => row.metrics).flatMap(row => [
                (['ScreenSpot-Pro', row.metrics!.screenspot_pro] as const),
                (['ScreenSpot-v2', row.metrics!.screenspot_v2] as const),
              ].map(([benchmark, metrics]) => <tr key={`${row.modelId}-${benchmark}`}>
                <td className={styles.modelName}>{row.modelName}</td><td>{benchmark}</td>
                <MetricCell value={metrics.by_ui_type?.text} /><MetricCell value={metrics.by_ui_type?.icon} />
                <MetricCell value={metrics} overall />
              </tr>))}</tbody>
            </table></div>
          </>}
          {currentMetrics && <div className={styles.details}>
            <details>
              <summary>查看 {current.modelName} 的应用级明细</summary>
              <div className={styles.tableWrap}><table>
                <thead><tr><th>Application</th><th>Benchmark</th><th>Accuracy</th><th>Hits / Samples</th></tr></thead>
                <tbody>{(['screenspot_pro', 'screenspot_v2'] as const).flatMap(key => {
                  const benchmark = key === 'screenspot_pro' ? 'ScreenSpot-Pro' : 'ScreenSpot-v2'
                  return Object.entries(applications(currentMetrics[key])).map(([app, value]) => <tr key={`${key}-${app}`}>
                    <td>{app.replaceAll('_', ' ')}</td><td>{benchmark}</td>
                    <td className={styles.metricCell}><strong>{percent(value.accuracy_percent)}</strong></td>
                    <td>{value.hits} / {value.samples}</td>
                  </tr>)
                })}</tbody>
              </table></div>
            </details>
          </div>}
          {current && <div className={styles.source}>数据源：{current.path}</div>}
        </div>
      </div>
    </section>
  )
}
