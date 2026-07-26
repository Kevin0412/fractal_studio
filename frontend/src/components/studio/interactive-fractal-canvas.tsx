"use client";

import { useRef } from "react";
import { LoaderCircle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FractalSpec } from "@/lib/api/platform";

type Props = {
  spec: FractalSpec;
  preview: string | null;
  previewing: boolean;
  width: number;
  height: number;
  onChange: (patch: Partial<FractalSpec>) => void;
  onReset: () => void;
};

export function InteractiveFractalCanvas({ spec, preview, previewing, width, height, onChange, onReset }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; re: number; im: number } | null>(null);

  const move = (x: number, y: number, baseRe: number, baseIm: number, startX: number, startY: number) => {
    const box = element.current?.getBoundingClientRect();
    if (!box) return;
    const scale = Number(spec.scale ?? 3);
    const aspect = box.width / box.height;
    onChange({ centerRe: baseRe - ((x - startX) / box.width) * scale * aspect, centerIm: baseIm + ((y - startY) / box.height) * scale });
  };

  return <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black shadow-2xl" ref={element}
    onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); drag.current = { x: event.clientX, y: event.clientY, re: Number(spec.centerRe ?? 0), im: Number(spec.centerIm ?? 0) }; }}
    onPointerMove={(event) => { const value = drag.current; if (value) move(event.clientX, event.clientY, value.re, value.im, value.x, value.y); }}
    onPointerUp={() => { drag.current = null; }}
    onWheel={(event) => {
      event.preventDefault();
      const box = element.current?.getBoundingClientRect(); if (!box) return;
      const oldScale = Number(spec.scale ?? 3); const nextScale = Math.min(1e9, Math.max(1e-12, oldScale * Math.exp(event.deltaY * 0.0015)));
      const x = (event.clientX - box.left) / box.width - 0.5; const y = (event.clientY - box.top) / box.height - 0.5; const aspect = box.width / box.height;
      const worldRe = Number(spec.centerRe ?? 0) + x * oldScale * aspect; const worldIm = Number(spec.centerIm ?? 0) - y * oldScale;
      onChange({ scale: nextScale, centerRe: worldRe - x * nextScale * aspect, centerIm: worldIm + y * nextScale });
    }}
    style={{ touchAction: "none", aspectRatio: `${width}/${height}` }}>
    {preview ? <img src={preview} alt="Fractal preview" draggable={false} className="h-full w-full select-none object-contain" /> : <div className="grid h-full min-h-[26rem] place-items-center text-sm text-muted-foreground">Generating first preview…</div>}
    <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/70 to-transparent p-3 text-xs text-white/70">
      <span>Drag pan · wheel zoom</span><span>{Number(spec.scale ?? 0).toExponential(3)} · {spec.iterations} it.</span>
    </div>
    {previewing && <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/25"><span className="flex items-center gap-2 rounded-full bg-black/70 px-3 py-1.5 text-sm"><LoaderCircle className="h-4 w-4 animate-spin" /> Rendering preview</span></div>}
    <Button size="sm" variant="secondary" className="absolute right-3 top-3" onClick={onReset}><RotateCcw className="mr-1 h-3.5 w-3.5" />Reset</Button>
  </div>;
}
