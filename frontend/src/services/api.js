const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function messageFor(error, fallback) {
  if (error.name === 'AbortError') return 'The analysis timed out. Check the service and try again.'
  return error.message || fallback
}

export async function predictImage(file, signal) {
  const body = new FormData()
  body.append('file', file)
  let response
  try {
    response = await fetch(`${API_URL}/api/predict?gradcam=true`, { method: 'POST', body, signal })
  } catch (error) {
    throw new Error(messageFor(error, 'The analysis service could not be reached.'))
  }
  let payload = null
  try { payload = await response.json() } catch { /* handled below */ }
  if (!response.ok) throw new Error(payload?.detail || 'The server returned an invalid response. Please try again.')
  if (!payload?.success || !['NORMAL', 'PNEUMONIA'].includes(payload.prediction)) throw new Error('The server response was incomplete. Please try again.')
  return payload
}

export async function getPerformance(signal) {
  const response = await fetch(`${API_URL}/api/performance`, { signal })
  if (!response.ok) throw new Error('Performance report unavailable')
  return response.json()
}
