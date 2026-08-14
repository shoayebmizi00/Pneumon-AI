import { Activity, Menu, X } from 'lucide-react'
import { useState } from 'react'

const links = [['Home','home'],['Detection','detection'],['How it works','how-it-works'],['Performance','performance'],['About','about']]

export default function Header() {
  const [open, setOpen] = useState(false)
  const go = (id) => { document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }); setOpen(false) }
  return <header className="sticky top-0 z-50 border-b border-teal-100/80 bg-mist/90 backdrop-blur-md">
    <nav className="mx-auto flex h-18 max-w-7xl items-center justify-between px-5 py-3 lg:px-8" aria-label="Primary navigation">
      <button onClick={() => go('home')} className="focus-ring flex min-h-11 items-center gap-3 rounded-xl" aria-label="PneumoAI home">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-teal-700 text-white shadow-card"><Activity size={21} aria-hidden="true" /></span>
        <span className="text-lg font-bold tracking-tight text-ink">Pneumo<span className="text-teal-600">AI</span></span>
      </button>
      <div className="hidden items-center gap-1 md:flex">{links.map(([label,id]) => <button key={id} onClick={() => go(id)} className="focus-ring min-h-11 rounded-lg px-3 text-sm font-medium text-slate-600 transition hover:bg-white hover:text-ink">{label}</button>)}</div>
      <button onClick={() => go('detection')} className="focus-ring hidden min-h-11 rounded-xl bg-teal-700 px-5 text-sm font-semibold text-white shadow-card transition hover:bg-teal-900 md:block">Try detection</button>
      <button onClick={() => setOpen(!open)} className="focus-ring grid h-11 w-11 place-items-center rounded-xl border border-teal-100 bg-white text-ink md:hidden" aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open}>{open ? <X/> : <Menu/>}</button>
    </nav>
    {open && <div className="border-t border-teal-100 bg-white px-5 py-3 md:hidden">{links.map(([label,id]) => <button key={id} onClick={() => go(id)} className="focus-ring block min-h-11 w-full rounded-lg px-3 text-left font-medium text-slate-700 hover:bg-teal-50">{label}</button>)}</div>}
  </header>
}
