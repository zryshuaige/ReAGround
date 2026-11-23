const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('reaground', {
  getModels: () => ipcRenderer.invoke('models'),
  getResults: () => ipcRenderer.invoke('results'),
  capture: () => ipcRenderer.invoke('capture'),
  predict: payload => ipcRenderer.invoke('predict', payload),
  cancel: predictionId => ipcRenderer.invoke('cancel', predictionId),
  click: predictionId => ipcRenderer.invoke('click', predictionId),
})
