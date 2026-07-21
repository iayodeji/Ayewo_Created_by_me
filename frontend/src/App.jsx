import { useMemo, useRef, useState } from 'react'

// Production uses the same FastAPI host for the UI and API. Vite proxies
// relative API requests to localhost:8000 during local frontend development.
const API_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const Icon = ({ name, size = 20 }) => {
  const paths = {
    activity: <><path d="M3 12h4l2.4-7 4.2 14 2.4-7H21" /></>,
    upload: <><path d="M12 16V3M7 8l5-5 5 5M5 21h14" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5M5 21h14" /></>,
    shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />,
    check: <path d="m5 12 4 4L19 6" />,
    alert: <><path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.8 3h15.6a2 2 0 0 0 1.8-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4M12 17h.01" /></>,
    x: <><path d="m18 6-12 12M6 6l12 12" /></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function ResultCard({ result, index }) {
  const positive = result.result?.toLowerCase().includes('parasit')
  const rawConfidence = Number(result.confidence) || 0
  const confidence = Math.round(rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence)
  return <article className={`result-card ${positive ? 'positive' : 'negative'}`}>
    <div className="result-top"><span className="sample">Sample {index + 1}</span><span className="status"><Icon name={positive ? 'alert' : 'check'} size={15} /> {positive ? 'Parasites detected' : 'No parasites detected'}</span></div>
    <div className="result-body">
      {result.preview && <img className="thumb" src={result.preview} alt="Uploaded blood smear" />}
      <div className="result-copy"><strong>{result.filename || result.result}</strong><span>{confidence}% model confidence</span>{result.low_confidence && <em>Manual review recommended</em>}</div>
      <div className="confidence"><div className="ring" style={{ '--progress': `${confidence * 3.6}deg` }}><b>{confidence}%</b></div></div>
    </div>
    {result.gradcam_image && <details><summary>View AI attention map</summary><img className="gradcam" src={`data:image/png;base64,${result.gradcam_image}`} alt="AI attention map" /></details>}
  </article>
}

export default function App() {
  const [mode, setMode] = useState('single')
  const [files, setFiles] = useState([])
  const [patient, setPatient] = useState({ code: '', feverDays: '', priority: false, vulnerable: false })
  const [results, setResults] = useState([])
  const [batchId, setBatchId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const maximum = mode === 'single' ? 1 : 50
  const inputFiles = useMemo(() => files.map(file => ({ file, preview: URL.createObjectURL(file) })), [files])
  function selectFiles(next) {
    const images = Array.from(next).filter(file => file.type.startsWith('image/'))
    if (!images.length) return setError('Please choose a PNG, JPG, or other image file.')
    setFiles(current => {
      const combined = mode === 'batch' ? [...current, ...images] : images
      return Array.from(new Map(combined.map(file => [`${file.name}-${file.size}-${file.lastModified}`, file])).values()).slice(0, maximum)
    }); setResults([]); setBatchId(null); setError('')
  }
  function switchMode(next) { setMode(next); setFiles([]); setResults([]); setBatchId(null); setError('') }
  async function screen() {
    if (!files.length) return setError('Add at least one blood smear image before screening.')
    setLoading(true); setError(''); setResults([]); setBatchId(null)
    const form = new FormData()
    files.forEach(file => form.append(mode === 'single' ? 'file' : 'files', file))
    try {
      const endpoint = mode === 'single' ? '/predict/single' : '/predict/batch?include_gradcam=false'
      const response = await fetch(`${API_URL}${endpoint}`, { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'The screening service could not process this request.')
      const returned = mode === 'single' ? [data] : data.results
      setResults(returned.map((item, index) => ({ ...item, filename: files[index]?.name, preview: inputFiles[index]?.preview })))
      setBatchId(data.batch_id || null)
    } catch (err) { setError(err.message || 'Unable to reach the API. Check that FastAPI is running.') }
    finally { setLoading(false) }
  }
  async function downloadReport() {
    if (!batchId) return
    const response = await fetch(`${API_URL}/report/${batchId}`)
    if (!response.ok) return setError('The report is no longer available. Run the batch again.')
    const url = URL.createObjectURL(await response.blob()); const link = document.createElement('a')
    link.href = url; link.download = `ayewo-report-${batchId}.pdf`; link.click(); URL.revokeObjectURL(url)
  }
  const positiveCount = results.filter(row => row.result?.toLowerCase().includes('parasit')).length
  const needsReview = results.some(row => row.low_confidence)
  const safetyFlag = patient.priority || patient.vulnerable
  const decisionTitle = safetyFlag ? 'Urgent clinical assessment' : positiveCount ? 'Parasite signal detected' : needsReview ? 'Confirm this screening result' : 'No parasite signal detected'
  const decisionText = safetyFlag ? 'A higher-risk patient flag was recorded. Do not delay the local clinical escalation pathway while this AI-assisted screen is reviewed.' : positiveCount ? 'Record the result and confirm it through your laboratory workflow before treatment decisions. Use local malaria case-management guidance.' : needsReview ? 'The model marked one or more images as uncertain. Check image quality and confirm with a quality-assured parasite test.' : 'No parasite signal was detected in these images. If malaria remains clinically suspected, follow local testing protocol and assess other causes of fever.'
  return <div className="app-shell">
    <header><a className="brand" href="#top"><span className="brand-mark"><Icon name="activity" /></span><span>ayewo<small>AI DIAGNOSTICS</small></span></a><div className="secure"><Icon name="shield" size={16} /> Secure clinical screening</div></header>
    <main id="top">
      <section className="hero"><div className="eyebrow">MALARIA SCREENING, REIMAGINED</div><h1>Clarity for every<br /><i>blood smear.</i></h1><p>Fast, AI-assisted malaria screening designed for the diagnostic realities of Nigerian laboratories.</p><div className="hero-metrics"><span><b>&lt; 1 min</b>screening time</span><span><b>Up to 50</b>samples per batch</span><span><b>75%+</b>review threshold</span></div></section>
      <section className="workspace" aria-label="Screening workspace"><div className="tabs" role="tablist"><button className={mode === 'single' ? 'active' : ''} onClick={() => switchMode('single')} role="tab"><Icon name="file" />Single sample</button><button className={mode === 'batch' ? 'active' : ''} onClick={() => switchMode('batch')} role="tab"><Icon name="activity" />Batch screening</button></div>
        <div className="screen-card"><div className="card-heading"><div><span className="step">01</span><h2>{mode === 'single' ? 'Screen a blood smear' : 'Screen a sample batch'}</h2><p>{mode === 'single' ? 'Upload one microscope image for an instant analysis.' : 'Upload up to 50 images for efficient laboratory screening.'}</p></div><span className="limit">{maximum === 1 ? '1 image' : 'MAX 50 IMAGES'}</span></div>
          <section className="patient-context" aria-labelledby="patient-context-title"><div><span className="context-label">OPTIONAL CLINICAL CONTEXT</span><h3 id="patient-context-title">Make the result easier to act on</h3></div><div className="patient-fields"><label>Patient / sample ID<input value={patient.code} onChange={event => setPatient(value => ({ ...value, code: event.target.value }))} placeholder="e.g. LAB-024" /></label><label>Days with fever<input type="number" min="0" value={patient.feverDays} onChange={event => setPatient(value => ({ ...value, feverDays: event.target.value }))} placeholder="e.g. 3" /></label></div><div className="clinical-flags"><label><input type="checkbox" checked={patient.priority} onChange={event => setPatient(value => ({ ...value, priority: event.target.checked }))} />Severe symptoms or clinician concern</label><label><input type="checkbox" checked={patient.vulnerable} onChange={event => setPatient(value => ({ ...value, vulnerable: event.target.checked }))} />Higher-risk patient (e.g. young child, pregnancy, immunocompromised)</label></div><p className="context-note">Stored only in this browser for the screening summary; it is not sent to the AI model.</p></section>
          <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); selectFiles(event.dataTransfer.files) }} onClick={() => inputRef.current.click()}><input ref={inputRef} type="file" accept="image/*" multiple={mode === 'batch'} onChange={event => selectFiles(event.target.files)} /><span className="upload-icon"><Icon name="upload" /></span><strong>Drop digital blood-smear images here</strong><p>from a microscope camera or compatible mobile capture adapter</p><small>PNG, JPG, WEBP · Maximum 10 MB per image</small></div>
          {files.length > 0 && <div className="file-strip">{files.map(file => <span key={`${file.name}-${file.lastModified}`}><Icon name="file" size={16} />{file.name}<button aria-label={`Remove ${file.name}`} onClick={() => setFiles(items => items.filter(item => item !== file))}><Icon name="x" size={14} /></button></span>)}</div>}
          {error && <div className="error" role="alert"><Icon name="alert" size={17} />{error}</div>}
          <button className="screen-button" onClick={screen} disabled={loading || !files.length}>{loading ? <><span className="spinner" />Analysing image{files.length > 1 ? 's' : ''}…</> : <><Icon name="activity" />Start AI screening</>}</button>
        </div>
      </section>
      {results.length > 0 && <section className="outcome"><div className="outcome-heading"><div><span className="step">02</span><h2>Screening results</h2><p>{positiveCount ? `${positiveCount} sample${positiveCount > 1 ? 's' : ''} require${positiveCount === 1 ? 's' : ''} attention.` : 'All screened samples show no parasite detection.'}</p></div>{batchId && <button className="report-button" onClick={downloadReport}><Icon name="download" size={18} />Download PDF report</button>}</div><aside className={`decision-brief ${safetyFlag || positiveCount ? 'attention' : needsReview ? 'review' : 'clear'}`} aria-live="polite"><div className="decision-icon"><Icon name={safetyFlag || positiveCount ? 'alert' : needsReview ? 'shield' : 'check'} /></div><div><span className="context-label">CLINICAL DECISION SUPPORT</span><h3>{decisionTitle}{patient.code && <small> · {patient.code}</small>}</h3><p>{decisionText}</p><div className="decision-tags">{patient.feverDays && <span>Fever: {patient.feverDays} day{patient.feverDays === '1' ? '' : 's'}</span>}{patient.priority && <span>Clinical concern flagged</span>}{patient.vulnerable && <span>Higher-risk patient</span>}{needsReview && <span>Manual review needed</span>}</div></div></aside><div className="results-grid">{results.map((result, index) => <ResultCard key={`${result.filename}-${index}`} result={result} index={index} />)}</div><div className="validation-note"><Icon name="shield" size={16} />Ayewo prioritizes digital smear review; it does not replace a validated malaria test, clinician judgment, or local emergency/referral pathways.</div></section>}
    </main>
    <footer><span>Ayewo AI Diagnostics</span><span>AI-assisted screening is a clinical decision-support tool. Confirm results through standard laboratory protocol.</span></footer>
  </div>
}
