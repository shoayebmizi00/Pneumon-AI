import { AlertCircle, CheckCircle2, FileImage, ImagePlus, LoaderCircle, RefreshCw, ShieldCheck, Trash2, UploadCloud } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { predictImage } from '../services/api'

const MAX_SIZE = 10 * 1024 * 1024
const ALLOWED = ['image/jpeg', 'image/png']
const percent = value => `${(Number(value) * 100).toFixed(1)}%`

function ScoreBar({ label, value, tone }) {
  return <div><div className="mb-2 flex justify-between text-sm"><span className="font-medium text-slate-700">{label}</span><span className="font-semibold tabular-nums text-ink">{percent(value)}</span></div><div className="h-2.5 overflow-hidden rounded-full bg-slate-100"><div className={`h-full rounded-full ${tone}`} style={{ width: percent(value) }} /></div></div>
}

function ResultCard({ result, preview }) {
  const positive = result.prediction === 'PNEUMONIA'
  const uncertain = result.uncertain
  return <div className="mt-6 space-y-5" aria-live="polite">
    <div className={`rounded-2xl border p-5 ${uncertain ? 'border-amber-200 bg-amber-50' : positive ? 'border-rose-200 bg-rose-50' : 'border-teal-200 bg-teal-50'}`}>
      <div className="flex items-start gap-3">{uncertain ? <AlertCircle className="mt-0.5 shrink-0 text-amber-600"/> : <CheckCircle2 className={`mt-0.5 shrink-0 ${positive ? 'text-rose-700' : 'text-teal-700'}`}/>}<div>
        <p className="text-xs font-bold uppercase tracking-[.16em] text-slate-500">AI prediction</p>
        <h3 className="mt-2 text-xl font-bold text-ink">{uncertain ? 'Uncertain AI result' : positive ? 'Pneumonia pattern detected' : 'No pneumonia pattern detected'}</h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">{uncertain ? 'The model does not have a strong in-domain prediction for this image. Professional medical review is strongly recommended.' : 'This model output is a research result, not a confirmed medical finding.'}</p>
        {result.domain_warning && <p className="mt-3 rounded-xl bg-white/70 p-3 text-sm font-medium text-amber-900">{result.domain_warning}</p>}
      </div></div>
    </div>
    <div className="rounded-2xl border border-slate-200 bg-white p-5">
      <div className="flex items-end justify-between border-b border-slate-100 pb-4"><div><p className="text-sm text-slate-500">Model confidence</p><p className="mt-1 text-3xl font-bold tabular-nums text-ink">{percent(result.prediction_score)}</p></div><p className="text-xs text-slate-500">Decision threshold {percent(result.decision_threshold)}</p></div>
      <div className="mt-5 space-y-4"><ScoreBar label="Normal score" value={result.normal_score} tone="bg-teal-600"/><ScoreBar label="Pneumonia score" value={result.pneumonia_score} tone="bg-rose-600"/></div>
    </div>
    {result.gradcam_image && <div className="rounded-2xl border border-slate-200 bg-white p-5"><h3 className="font-bold text-ink">Visual comparison</h3><div className="mt-4 grid gap-4 sm:grid-cols-2"><figure><img src={preview} alt="Original uploaded chest X-ray" className="aspect-square w-full rounded-xl bg-slate-950 object-contain"/><figcaption className="mt-2 text-center text-sm font-medium text-slate-600">Original X-ray</figcaption></figure><figure><img src={result.gradcam_image} alt="Grad-CAM attention map overlaid on the X-ray" className="aspect-square w-full rounded-xl bg-slate-950 object-contain"/><figcaption className="mt-2 text-center text-sm font-medium text-slate-600">AI attention map</figcaption></figure></div><p className="mt-4 text-sm leading-6 text-slate-600">The attention map visualizes regions that influenced the AI prediction. It is not a medical diagnosis or lesion localization.</p></div>}
  </div>
}

export default function Detection() {
  const [file, setFile] = useState(null), [preview, setPreview] = useState(''), [result, setResult] = useState(null)
  const [error, setError] = useState(''), [loading, setLoading] = useState(false), [dragging, setDragging] = useState(false)
  const inputRef = useRef(null), abortRef = useRef(null)
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); abortRef.current?.abort() }, [preview])
  const choose = candidate => {
    setError(''); setResult(null)
    if (!candidate) return
    if (!ALLOWED.includes(candidate.type)) return setError('Unsupported file. Choose a JPG, JPEG, or PNG chest X-ray.')
    if (candidate.size > MAX_SIZE) return setError('The image is larger than 10 MB. Choose a smaller file.')
    if (preview) URL.revokeObjectURL(preview)
    setFile(candidate); setPreview(URL.createObjectURL(candidate))
  }
  const remove = () => { abortRef.current?.abort(); if (preview) URL.revokeObjectURL(preview); setFile(null); setPreview(''); setResult(null); setError(''); if (inputRef.current) inputRef.current.value = '' }
  const analyze = async () => {
    if (!file) return setError('Choose a chest X-ray image before analysis.')
    setLoading(true); setError(''); setResult(null)
    const controller = new AbortController(); abortRef.current = controller
    const timeout = setTimeout(() => controller.abort(), 45000)
    try { setResult(await predictImage(file, controller.signal)) }
    catch (err) { setError(err.message) }
    finally { clearTimeout(timeout); setLoading(false) }
  }
  return <section id="detection" className="section-pad bg-mist"><div className="mx-auto max-w-7xl px-5 lg:px-8">
    <div className="section-heading"><span className="eyebrow">Pneumonia detection</span><h2>Upload one chest X-ray for research analysis</h2><p>Images are processed in memory for this request and are not intentionally retained by the application.</p></div>
    <div className="mx-auto mt-10 grid max-w-5xl gap-6 lg:grid-cols-[1.05fr_.95fr]">
      <div className="rounded-3xl border border-teal-100 bg-white p-5 shadow-soft sm:p-7">
        {!file ? <div onDragEnter={e => {e.preventDefault();setDragging(true)}} onDragOver={e=>e.preventDefault()} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);choose(e.dataTransfer.files[0])}} className={`grid min-h-80 place-items-center rounded-2xl border-2 border-dashed p-8 text-center transition ${dragging ? 'border-teal-600 bg-teal-50' : 'border-teal-200 bg-mist'}`}>
          <div><span className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-teal-100 text-teal-700"><UploadCloud size={28}/></span><h3 className="mt-5 text-xl font-bold text-ink">Upload chest X-ray</h3><p className="mt-2 text-sm leading-6 text-slate-600">Drag and drop here, or select an image. JPG or PNG, up to 10 MB.</p><button onClick={()=>inputRef.current?.click()} className="focus-ring mt-5 inline-flex min-h-11 items-center gap-2 rounded-xl bg-teal-700 px-5 font-semibold text-white hover:bg-teal-900"><ImagePlus size={18}/> Browse image</button><input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={e=>choose(e.target.files[0])} className="sr-only" aria-label="Choose chest X-ray image"/></div>
        </div> : <div><div className="relative overflow-hidden rounded-2xl bg-slate-950"><img src={preview} alt="Selected chest X-ray preview" className="aspect-[4/3] w-full object-contain"/><button onClick={remove} disabled={loading} className="focus-ring absolute right-3 top-3 grid h-11 w-11 place-items-center rounded-xl bg-white/95 text-rose-700 shadow-card hover:bg-white disabled:opacity-50" aria-label="Remove selected image"><Trash2 size={19}/></button></div><div className="mt-4 flex items-center gap-3"><FileImage className="shrink-0 text-teal-700"/><div className="min-w-0"><p className="truncate text-sm font-semibold text-ink">{file.name}</p><p className="text-xs text-slate-500">{(file.size/1024/1024).toFixed(2)} MB</p></div></div><button onClick={analyze} disabled={loading} className="focus-ring mt-5 flex min-h-12 w-full items-center justify-center gap-2 rounded-xl bg-teal-700 px-5 font-semibold text-white hover:bg-teal-900 disabled:cursor-not-allowed disabled:opacity-60">{loading ? <><LoaderCircle className="animate-spin" size={19}/> Analyzing X-ray…</> : <><ShieldCheck size={19}/> Analyze X-ray</>}</button>{loading && <p className="mt-3 text-center text-sm text-slate-500">The AI model is processing the uploaded image.</p>}</div>}
        {error && <div role="alert" className="mt-4 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"><AlertCircle className="mt-0.5 shrink-0" size={18}/><div><p>{error}</p>{file && <button onClick={analyze} className="focus-ring mt-2 inline-flex min-h-11 items-center gap-2 rounded-lg font-semibold underline"><RefreshCw size={16}/> Retry analysis</button>}</div></div>}
      </div>
      <div>{result ? <ResultCard result={result} preview={preview}/> : <div className="h-full rounded-3xl border border-slate-200 bg-white p-7 shadow-card"><span className="grid h-12 w-12 place-items-center rounded-xl bg-slate-100 text-slate-600"><ShieldCheck/></span><h3 className="mt-5 text-xl font-bold text-ink">What you’ll receive</h3><ul className="mt-5 space-y-4 text-sm leading-6 text-slate-600">{['A NORMAL or PNEUMONIA pattern classification','Separate model scores and the selected threshold','An explicit warning for borderline or out-of-domain results','A Grad-CAM attention map when model inference is available'].map(item=><li key={item} className="flex gap-3"><CheckCircle2 size={18} className="mt-1 shrink-0 text-teal-600"/>{item}</li>)}</ul><div className="mt-7 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900"><strong>Medical disclaimer:</strong> This AI system is an educational/research prototype and does not provide a medical diagnosis. Predictions may be incorrect.</div></div>}</div>
    </div>
  </div></section>
}
