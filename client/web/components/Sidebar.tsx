import { ChartIcon, DemoIcon, PanelIcon } from './Icons'
import logoMark from '../../assets/2.png'
import styles from './Sidebar.module.css'

interface Props {
  page: 'demo' | 'evaluation'
  collapsed: boolean
  onCollapse(): void
  onNavigate(page: 'demo' | 'evaluation'): void
}

export default function Sidebar({ page, collapsed, onCollapse, onNavigate }: Props) {
  return (
    <aside className={`${styles.sidebar} ${collapsed ? styles.collapsed : ''}`}>
      <div className={styles.top}>
        <div className={styles.brand}>
          <img className={styles.logo} src={logoMark} alt="ReAGround" />
          {!collapsed && <div><strong>ReAGround</strong><span>GUI GROUNDING</span></div>}
        </div>
        <button className={styles.collapse} onClick={onCollapse} aria-label={collapsed ? '展开侧边栏' : '折叠侧边栏'} title={collapsed ? '展开侧边栏' : '折叠侧边栏'}><PanelIcon /></button>
      </div>
      <nav className={styles.nav} aria-label="主导航">
        <span className={styles.section}>{collapsed ? '' : '工作区'}</span>
        <button className={`${styles.navItem} ${page === 'demo' ? styles.active : ''}`} onClick={() => onNavigate('demo')} title="Demo" aria-current={page === 'demo' ? 'page' : undefined}>
          <DemoIcon />{!collapsed && <span>Demo</span>}
        </button>
        <button className={`${styles.navItem} ${page === 'evaluation' ? styles.active : ''}`} onClick={() => onNavigate('evaluation')} title="模型评测" aria-current={page === 'evaluation' ? 'page' : undefined}>
          <ChartIcon />{!collapsed && <span>模型评测</span>}
        </button>
      </nav>
      <div className={styles.footer} title="模型与数据均在本机运行">
        <span className={styles.localDot} />{!collapsed && <span>本地运行</span>}
      </div>
    </aside>
  )
}
