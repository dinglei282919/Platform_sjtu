export type Group = 'normal' | 'fault' | 'scsc' | 'ccsc'
export type Method = 'scsc' | 'ccsc'
export type Output = 'C' | 'T' | 'Tc'
export type Signal = Output | 'Ci' | 'Ti' | 'Tci' | 'Qc'
export type Config = {
  scenario: string; mode: string; widths: number[]; references: number[]; scsc_interval: number; ccsc_interval: number
  scsc_gain: number; ccsc_gain: number; speed: number; seed: number
}
export type Options = {
  groups: Record<Group,string>; scenarios: Record<string,string>; modes: Record<string,string>
  signals: Record<Signal,[string,string]>; nominal: number[]; flow_limits: number[]
  period: number; fault_mode: 'manual'; post_fault_min: number; defaults: Config; presets: Record<string,Record<string,Config>>
  trigger_limits: Record<Output, {reference: [number,number]; half_width: [number,number]}>
  parameters: Record<'interval'|'gain'|'seed'|'speed', {min:number; max:number; step:number; decimals:number; hint:string}>
}
type Branch = Record<Signal,number> & {outside: Output[]; a: number; b: number}
export type Compensation = {group: Group; at: number; channels: Output[]; error: number[]; first_seq: number; first_sample_min: number}
export type Row = {run: string; seq: number; t: number; groups: Record<Group,Branch>; compensation: Compensation[]}
export type Snapshot = {
  id: string; run: string; seq: number; status: string; error: string; resync: boolean; latest: Row
  fault_min: number | null
  effect: {signal: 'C' | 'T'; points: [number, number, number, number][]}
  config: Config; samples: Row[]; recent: (Compensation & {label: string})[]
  mapping: {version: string; seed: number; samples: number} | null
  metrics: {groups: Record<Group,{first_update: number | null}>}
  view: {state: string; status: string; speed: string; timing: string[]; headings: string[]; recovery_help:string
         effect: {phase: string; instruction: string; title: string; reference: string; unit: string; domain: [number,number]
                  methods: Record<Method,{first_min: number | null; step: string; readout: string; limits: [number,number]; tick: number}>}
         metrics: string[][]; diagrams: Record<Group,Record<string,string>>}
}
export const outputs: Output[] = ['C','T','Tc']
export const healingColors = ['#84b8d9','#f5bd73','#6cd6ae','#c6a6ff','#889eb4','#889eb4']
export const inputs: Signal[] = ['Ci','Ti','Tci','Qc']
