import { ArrowRight, BrainCircuit, CheckCircle2, Microscope } from 'lucide-react'

export default function Hero() {
  const go = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  return <section id="home" className="relative overflow-hidden border-b border-teal-100 bg-[radial-gradient(circle_at_85%_15%,#d8f1ec_0,transparent_36%)]">
    <div className="mx-auto grid min-h-[640px] max-w-7xl items-center gap-12 px-5 py-20 lg:grid-cols-[1.1fr_.9fr] lg:px-8">
      <div>
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-teal-200 bg-white px-3 py-2 text-sm font-semibold text-teal-700 shadow-card"><Microscope size={16}/> Research & education prototype</div>
        <h1 className="max-w-3xl text-4xl font-bold leading-[1.08] tracking-[-.035em] text-ink sm:text-5xl lg:text-6xl">A careful second look at chest X-rays, powered by AI.</h1>
        <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">Upload a chest X-ray to explore a deep-learning model’s prediction, calibrated score, and attention map—without presenting the result as a diagnosis.</p>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <button onClick={() => go('detection')} className="focus-ring inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-teal-700 px-6 font-semibold text-white shadow-soft transition hover:bg-teal-900">Analyze an X-ray <ArrowRight size={18}/></button>
          <button onClick={() => go('how-it-works')} className="focus-ring min-h-12 rounded-xl border border-teal-200 bg-white px-6 font-semibold text-ink transition hover:border-teal-500">Learn how it works</button>
        </div>
        <div className="mt-8 flex flex-wrap gap-x-6 gap-y-3 text-sm text-slate-600">{['No image retention','Patient-grouped evaluation','Explicit uncertainty'].map(item => <span key={item} className="flex items-center gap-2"><CheckCircle2 size={16} className="text-teal-600"/>{item}</span>)}</div>
      </div>
      <div className="relative mx-auto w-full max-w-lg" aria-hidden="true">
        <div className="absolute -inset-6 rounded-[2.5rem] bg-teal-100/60 blur-2xl" />
        <div className="relative overflow-hidden rounded-[2rem] border border-white/80 bg-ink p-5 shadow-soft">
          <div className="flex items-center justify-between border-b border-white/10 pb-4 text-white"><span className="text-sm font-semibold">Research inference</span><span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_0_6px_rgba(52,211,153,.12)]"/></div>
          <div className="mt-5 grid aspect-[4/3] place-items-center overflow-hidden rounded-2xl bg-[radial-gradient(ellipse_at_center,#47706c_0,#183f3c_40%,#091f20_72%)]">
            <div className="relative h-4/5 w-3/5 rounded-[48%_48%_40%_40%] border border-white/20 bg-white/5"><div className="absolute left-1/2 top-0 h-full w-px bg-white/15"/><div className="absolute left-[14%] top-[18%] h-[65%] w-[31%] rounded-[50%] border border-white/20 bg-black/15"/><div className="absolute right-[14%] top-[18%] h-[65%] w-[31%] rounded-[50%] border border-white/20 bg-black/15"/><BrainCircuit className="absolute bottom-5 left-1/2 -translate-x-1/2 text-teal-200" size={32}/></div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3"><div className="rounded-xl bg-white/10 p-3"><div className="text-xs text-teal-100">Output</div><div className="mt-1 text-sm font-semibold text-white">Pattern screening</div></div><div className="rounded-xl bg-white/10 p-3"><div className="text-xs text-teal-100">Explainability</div><div className="mt-1 text-sm font-semibold text-white">Grad-CAM</div></div></div>
        </div>
      </div>
    </div>
  </section>
}
