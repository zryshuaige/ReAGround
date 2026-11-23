const { app, BrowserWindow, dialog, ipcMain, Menu, net, protocol } = require('electron')
const { spawn } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { pathToFileURL } = require('node:url')

app.disableHardwareAcceleration()
protocol.registerSchemesAsPrivileged([{
  scheme: 'reaground-image',
  privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true },
}])

let mainWindow = null
let sidecar = null
let sidecarConfig = null
let sidecarReady = null
let quitting = false
let shutdownStarted = false

function findWorkspace() {
  const starts = [__dirname, path.dirname(app.getPath('exe')), process.cwd()]
  for (const start of starts) {
    let current = path.resolve(start)
    while (true) {
      if (fs.existsSync(path.join(current, 'models', 'Qwen2.5-VL-3B-Instruct'))
        && fs.existsSync(path.join(current, 'client', 'models.json'))) return current
      const parent = path.dirname(current)
      if (parent === current) break
      current = parent
    }
  }
  throw new Error('找不到包含 models 与 client/models.json 的项目目录')
}

function startSidecar() {
  return new Promise((resolve, reject) => {
    const workspace = findWorkspace()
    const packed = app.isPackaged
    const command = packed
      ? path.join(process.resourcesPath, 'sidecar', 'reaground-sidecar.exe')
      : (process.env.REAGROUND_PYTHON || 'python')
    const args = packed ? [] : [path.join(workspace, 'client', 'sidecar', 'app.py')]
    sidecar = spawn(command, args, {
      cwd: workspace,
      windowsHide: true,
      env: { ...process.env, PYTHONUTF8: '1', REAGROUND_WORKSPACE: workspace },
    })
    let stdout = ''
    let stderr = ''
    const timeout = setTimeout(() => reject(new Error('Python sidecar 启动超时')), 120000)
    sidecar.stdout.on('data', chunk => {
      stdout += chunk.toString()
      const lines = stdout.split(/\r?\n/)
      stdout = lines.pop() || ''
      for (const line of lines) {
        if (!line.trimStart().startsWith('{')) continue
        try {
          const ready = JSON.parse(line)
          if (!ready.port || !ready.token) continue
          sidecarConfig = { port: ready.port, token: ready.token }
          clearTimeout(timeout)
          resolve(sidecarConfig)
        } catch {}
      }
    })
    sidecar.stderr.on('data', chunk => {
      stderr = (stderr + chunk.toString()).slice(-12000)
      console.error('[sidecar]', chunk.toString().trim())
    })
    sidecar.on('error', error => {
      clearTimeout(timeout)
      reject(error)
    })
    sidecar.on('exit', code => {
      if (!quitting) {
        const detail = stderr ? `\n${stderr.slice(-1500)}` : ''
        dialog.showErrorBox('Python sidecar 已退出', `退出码：${code}${detail}`)
      }
    })
  })
}

async function api(route, options = {}) {
  if (!sidecarConfig) await sidecarReady
  const response = await fetch(`http://127.0.0.1:${sidecarConfig.port}${route}`, {
    method: options.method || 'GET',
    headers: {
      'X-ReAGround-Token': sidecarConfig.token,
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.error || `请求失败：${response.status}`)
  return payload
}

function registerIpc() {
  ipcMain.handle('models', () => api('/api/models'))
  ipcMain.handle('results', () => api('/api/results'))
  ipcMain.handle('capture', async () => {
    if (!mainWindow) throw new Error('窗口尚未就绪')
    mainWindow.hide()
    await new Promise(resolve => setTimeout(resolve, 350))
    try {
      return await api('/api/capture', { method: 'POST', body: {} })
    } finally {
      mainWindow.show()
      mainWindow.focus()
    }
  })
  ipcMain.handle('predict', (_, payload) => api('/api/predict', { method: 'POST', body: payload }))
  ipcMain.handle('cancel', (_, predictionId) => api('/api/cancel', {
    method: 'POST', body: { predictionId },
  }))
  ipcMain.handle('click', (_, predictionId) => api('/api/click', {
    method: 'POST', body: { predictionId },
  }))
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1240,
    height: 860,
    minWidth: 1000,
    minHeight: 720,
    show: false,
    backgroundColor: '#f7f8fa',
    title: 'ReAGround',
    icon: path.join(__dirname, '..', 'assets', 'reaground.ico'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })
  mainWindow.once('ready-to-show', () => mainWindow.show())
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  mainWindow.webContents.on('will-navigate', event => event.preventDefault())
  mainWindow.loadFile(path.join(__dirname, '..', 'web-dist', 'index.html'))
}

app.whenReady().then(() => {
  protocol.handle('reaground-image', request => {
    const url = new URL(request.url)
    const requested = path.resolve(url.searchParams.get('path') || '')
    const tempRoot = path.resolve(os.tmpdir()) + path.sep
    const serviceRoot = path.join(path.dirname(requested))
    if (!requested.startsWith(tempRoot)
      || !path.basename(serviceRoot).match(/^[0-9a-f]{32}$/)
      || !['capture.png', 'prediction.png'].includes(path.basename(requested))) {
      throw new Error('非法图片路径')
    }
    return net.fetch(pathToFileURL(requested).toString())
  })
  registerIpc()
  Menu.setApplicationMenu(null)
  createWindow()
  sidecarReady = startSidecar()
  sidecarReady.catch(error => {
    dialog.showErrorBox('无法启动模型服务', error.message)
  })
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => app.quit())
app.on('before-quit', event => {
  if (shutdownStarted || !sidecar) return
  event.preventDefault()
  shutdownStarted = true
  quitting = true
  const finish = () => {
    if (sidecar) sidecar.kill()
    app.quit()
  }
  if (!sidecarConfig) {
    finish()
    return
  }
  api('/api/shutdown', { method: 'POST', body: {} })
    .catch(() => {})
    .finally(() => setTimeout(finish, 300))
})
