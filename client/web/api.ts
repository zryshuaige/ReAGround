export const api = window.reaground

export function imageUrl(path: string): string {
  return `reaground-image://local/?path=${encodeURIComponent(path)}`
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
