import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

afterEach(() => vi.restoreAllMocks())

describe('PneumoAI interface', () => {
  it('shows safety language and upload control', () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('offline'))
    render(<App />)
    expect(screen.getByRole('heading', { name: /careful second look/i })).toBeInTheDocument()
    expect(screen.getAllByText(/does not provide a medical diagnosis/i).length).toBeGreaterThan(0)
    expect(document.querySelector('input[type="file"]')).toBeInTheDocument()
  })

  it('rejects unsupported files before a request', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('offline'))
    render(<App />)
    const input = document.querySelector('input[type="file"]')
    await userEvent.upload(input, new File(['not an image'], 'report.pdf', { type: 'application/pdf' }), { applyAccept: false })
    expect(screen.getByRole('alert')).toHaveTextContent(/unsupported file/i)
  })
})
