import Header from './components/Header'
import Hero from './components/Hero'
import Detection from './components/Detection'
import { About, Disclaimer, HowItWorks, Performance } from './components/InformationSections'

export default function App() { return <><Header/><main id="main" tabIndex="-1"><Hero/><Detection/><HowItWorks/><Performance/><About/><Disclaimer/></main><footer className="bg-white"><div className="mx-auto flex max-w-7xl flex-col gap-2 px-5 py-8 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between lg:px-8"><p>© 2026 PneumoAI · University research prototype</p><a href="#disclaimer" className="focus-ring min-h-11 content-center rounded-lg font-medium text-teal-700 underline-offset-4 hover:underline">Medical disclaimer</a></div></footer></> }
