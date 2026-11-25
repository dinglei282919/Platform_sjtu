import {useId, useLayoutEffect, useRef} from 'react'

export function ErrorDialog({title, message, onClose}: {title:string; message:string; onClose:()=>void}) {
  const ref=useRef<HTMLDialogElement>(null)
  const id=useId()
  useLayoutEffect(()=>{
    const dialog=ref.current!
    dialog.showModal()
    return ()=>dialog.close()
  },[])
  return <dialog ref={ref} className="anomaly-error-dialog" role="alertdialog" aria-modal="true"
    aria-labelledby={id+'-title'} aria-describedby={id+'-message'}
    onKeyDown={event=>{
      event.stopPropagation()
      if(event.key==='Escape'){event.preventDefault();onClose()}
    }} onCancel={event=>{event.preventDefault();onClose()}}>
    <header><h3 id={id+'-title'}>{title}</h3>
      <button type="button" className="anomaly-dialog-close" aria-label="关闭错误弹窗" onClick={onClose}>×</button></header>
    <div id={id+'-message'} className="anomaly-dialog-message">{message}</div>
    <footer><button type="button" autoFocus onClick={onClose}>确定</button></footer>
  </dialog>
}
